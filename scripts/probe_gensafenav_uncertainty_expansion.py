#!/usr/bin/env python3
"""Frozen CPU native episode plus same-state conservative-input shadow actions."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args, out, summary, protocol):
    # Do not import torch before hiding CUDA. This experiment has no GPU path.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    import probe_gensafenav_feedback_selection as original

    vendor = ROOT / "vendor/GenSafeNav"
    os.chdir(vendor)
    sys.path.insert(0, str(vendor))
    import rl.networks.envs as native_env_module
    from rl.networks.model import Policy

    def rng_state():
        return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())

    def restore_rng(state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])

    def swept(robot_start, robot_end, human_start, human_end, radii):
        r0 = np.asarray(robot_start, dtype=np.float64) - np.asarray(human_start, dtype=np.float64)
        r1 = np.asarray(robot_end, dtype=np.float64) - np.asarray(human_end, dtype=np.float64)
        direction = r1 - r0
        denominator = float(direction @ direction)
        fraction = float(np.clip(-(r0 @ direction) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
        return float(np.linalg.norm(r0 + fraction * direction) - radii), fraction

    native_act = Policy.act
    native_factory = native_env_module.make_vec_envs
    pending = {}
    base_holder = {}
    native_rows, action_rows = [], []
    entity_refs, entity_keys = [], {}
    max_endpoint_error = 0.0

    def lifetime(human):
        key = id(human)
        if key not in entity_keys:
            entity_keys[key] = len(entity_refs)
            entity_refs.append(human)
        return entity_keys[key]

    def observed_act(self, inputs, hidden, masks, deterministic=False):
        assert deterministic and not pending, "Expected exactly one paired actor call per physical transition"
        assert summary["extra_activity"]["shadow_actor_calls"] < protocol["fixed_run"]["max_shadow_actor_calls"]
        shadow_inputs = {name: value.detach().clone() for name, value in inputs.items()}
        shadow_hidden = {name: value.detach().clone() for name, value in hidden.items()}
        shadow_masks = masks.detach().clone()
        original_q = inputs["conformity_scores"].detach().clone()
        assert original_q.shape[-1] == 5 and bool(torch.isfinite(original_q).all())
        assert bool((original_q >= 0).all()) and bool((original_q <= 1).all())
        assert self.base.args.sort_humans
        detected_count = inputs["detected_human_num"].reshape(original_q.shape[0], -1)
        assert detected_count.shape[-1] == 1
        native_prefix_mask = (torch.arange(original_q.shape[-2], device=original_q.device).unsqueeze(0)
                              < detected_count.int())
        eligible = native_prefix_mask & (original_q > 0).any(dim=-1)
        assert eligible.shape == original_q.shape[:-1]
        expanded_q = torch.where(eligible.unsqueeze(-1),
                                 torch.clamp(original_q + protocol["fixed_run"]["delta_meters"], max=1.0), original_q)
        assert bool((expanded_q >= original_q).all())
        assert torch.equal(expanded_q[~eligible], original_q[~eligible])
        shadow_inputs["conformity_scores"] = expanded_q
        before_rng = rng_state()
        before_buffers = {name: value.detach().clone() for name, value in self.named_buffers()}
        native_result = native_act(self, inputs, hidden, masks, deterministic=deterministic)
        after_rng = rng_state()
        after_buffers = {name: value.detach().clone() for name, value in self.named_buffers()}
        try:
            restore_rng(before_rng)
            with torch.no_grad():
                for name, value in self.named_buffers():
                    value.copy_(before_buffers[name])
            summary["extra_activity"]["shadow_actor_calls"] += 1
            shadow_result = native_act(self, shadow_inputs, shadow_hidden, shadow_masks, deterministic=deterministic)
            summary["extra_activity"]["shadow_actor_calls_completed"] += 1
        finally:
            restore_rng(after_rng)
            with torch.no_grad():
                for name, value in self.named_buffers():
                    value.copy_(after_buffers[name])
        assert torch.equal(inputs["conformity_scores"], original_q), "Original uncertainty input was modified"
        changes = expanded_q - original_q
        pending.update(
            native_nominal=native_result[1].detach().cpu().numpy()[0].copy(),
            shadow_nominal=shadow_result[1].detach().cpu().numpy()[0].copy(),
            q={"original": original_q.cpu().tolist(), "expanded": expanded_q.cpu().tolist(),
               "min_delta": float(changes.min()), "max_delta": float(changes.max()),
               "changed_count": int((changes > 0).sum()), "total_count": int(changes.numel()),
               "eligible_rows": eligible.cpu().tolist(), "eligible_row_count": int(eligible.sum()),
               "native_detected_count": detected_count.cpu().tolist(),
               "native_prefix_mask": native_prefix_mask.cpu().tolist(),
               "nesting_passed": True})
        return native_result

    def observed_factory(*a, **kw):
        envs = native_factory(*a, **kw)
        base = envs.venv.unwrapped.envs[0].env
        assert base.robot.visible is False and base.robot.kinematics == "holonomic"
        assert base.robot.policy.name == "selfAttn_merge_srnn"
        base_holder["base"] = base
        native_step = base.step

        def observed_step(*step_args, **step_kw):
            nonlocal max_endpoint_error
            assert pending and len(native_rows) < protocol["fixed_run"]["max_physical_steps"]
            robot = base.robot
            robot_start = np.asarray(robot.get_position(), dtype=np.float64)
            robot_radius = float(robot.radius)
            goal = np.asarray(robot.get_goal_position(), dtype=np.float64)
            assert robot.time_step == base.time_step == 0.25
            assert np.isfinite(robot_start).all() and np.isfinite(goal).all() and np.isfinite(robot_radius)
            humans = list(base.humans)
            human_before = []
            for h in humans:
                assert h.kinematics == "holonomic"
                assert h.time_step == base.time_step
                assert np.isfinite(h.get_position()).all() and np.isfinite(h.radius)
                human_before.append({"lifetime": lifetime(h), "slot": int(h.id),
                                     "position": np.asarray(h.get_position(), dtype=np.float64),
                                     "radius": float(h.radius)})
            passed_action = np.asarray(step_args[0] if step_args else step_kw["action"]).copy()
            assert np.array_equal(passed_action, pending["native_nominal"])
            native_control = robot.policy.clip_action(passed_action.copy(), robot.v_pref)
            shadow_control = robot.policy.clip_action(pending["shadow_nominal"].copy(), robot.v_pref)
            native_endpoint = np.asarray(robot.compute_position(native_control, base.time_step), dtype=np.float64)
            shadow_endpoint = np.asarray(robot.compute_position(shadow_control, base.time_step), dtype=np.float64)
            assert np.isfinite(native_endpoint).all() and np.isfinite(shadow_endpoint).all()
            result = native_step(*step_args, **step_kw)
            robot_end = np.asarray(robot.get_position(), dtype=np.float64)
            assert np.isfinite(robot_end).all()
            error = float(np.max(np.abs(robot_end - native_endpoint)))
            max_endpoint_error = max(max_endpoint_error, error)
            assert error <= 1e-8, f"Native decoded endpoint mismatch: {error}"
            assert float(robot.radius) == robot_radius
            old_ids = {id(h) for h in humans}
            current_ids = {id(h) for h in base.humans}
            pairs = []
            for h, before in zip(humans, human_before):
                assert float(h.radius) == before["radius"]
                human_end = np.asarray(h.get_position(), dtype=np.float64)
                assert np.isfinite(human_end).all()
                radii = robot_radius + before["radius"]
                c_native, tau_native = swept(robot_start, robot_end, before["position"], human_end, radii)
                c_shadow, tau_shadow = swept(robot_start, shadow_endpoint, before["position"], human_end, radii)
                c_braking, tau_braking = swept(robot_start, robot_start, before["position"], human_end, radii)
                pairs.append({"lifetime_id": before["lifetime"], "human_id": before["slot"],
                              "human_radius": before["radius"], "start": before["position"].tolist(),
                              "end_of_actual_motion": human_end.tolist(), "replaced_after_motion": id(h) not in current_ids,
                              "native_swept_clearance": c_native, "shadow_swept_clearance": c_shadow,
                              "braking_swept_clearance": c_braking,
                              "initial_clearance": float(np.linalg.norm(robot_start - before["position"]) - radii),
                              "native_endpoint_clearance": float(np.linalg.norm(robot_end - human_end) - radii),
                              "shadow_endpoint_clearance": float(np.linalg.norm(shadow_endpoint - human_end) - radii),
                              "native_minimizer_fraction": tau_native, "shadow_minimizer_fraction": tau_shadow,
                              "braking_minimizer_fraction": tau_braking})
            native_min = min(p["native_swept_clearance"] for p in pairs)
            shadow_min = min(p["shadow_swept_clearance"] for p in pairs)
            braking_min = min(p["braking_swept_clearance"] for p in pairs)
            row = {"physical_index_before": int(base.step_counter) - 1,
                   "robot_start": robot_start.tolist(), "robot_actual_end": robot_end.tolist(),
                   "robot_shadow_end": shadow_endpoint.tolist(), "robot_radius": robot_radius,
                   "native_control": list(map(float, native_control)), "shadow_control": list(map(float, shadow_control)),
                   "native_swept_clearance": native_min, "shadow_swept_clearance": shadow_min,
                   "braking_swept_clearance": braking_min, "braking_control": [0.0, 0.0],
                   "clearance_difference_shadow_minus_native": shadow_min - native_min,
                   "native_goal_progress": float(np.linalg.norm(goal - robot_start) - np.linalg.norm(goal - robot_end)),
                   "shadow_goal_progress": float(np.linalg.norm(goal - robot_start) - np.linalg.norm(goal - shadow_endpoint)),
                   "braking_goal_progress": 0.0,
                   "new_swept_overlap": native_min >= 0 and shadow_min < 0,
                   "replacements_after_motion": len(current_ids - old_ids), "human_pairs": pairs,
                   "native_endpoint_error": error}
            native_rows.append(row)
            action_rows.append({"physical_index_before": row["physical_index_before"],
                                "native_nominal": pending["native_nominal"].tolist(),
                                "shadow_nominal": pending["shadow_nominal"].tolist(), "radii": pending["q"]})
            for filename, data in [("paired_physical_consequences.jsonl", row), ("paired_actor_inputs.jsonl", action_rows[-1])]:
                with (out / filename).open("a") as handle:
                    handle.write(json.dumps(data, allow_nan=False) + "\n")
            pending.clear()
            return result

        base.step = observed_step
        return envs

    native_summary = {
        "status": "starting", "_started": summary["_started"],
        "activity": {name: 0 for name in (
            "actor_calls", "gst_calls", "physical_steps", "physical_resets", "actor_calls_completed",
            "gst_calls_completed", "physical_steps_completed", "checkpoint_loads", "checkpoint_loads_completed",
            "actual_robot_controls", "talk2env_calls", "post_terminal_talk2env_calls")}}
    native_out = out / "native"
    native_out.mkdir()
    Policy.act, native_env_module.make_vec_envs = observed_act, observed_factory
    try:
        # Calling run directly intentionally skips the old main's whole-resource hash sweep.
        original.run(SimpleNamespace(vendor=vendor, device="cpu", wall_seconds=120), native_out, native_summary)
    finally:
        Policy.act, native_env_module.make_vec_envs = native_act, native_factory
        native_summary.pop("_started", None)
        write_json(native_out / "summary.json", native_summary)
        summary["native"] = native_summary
        summary["physical_pairs_completed"] = len(native_rows)
        summary["native_endpoint_max_error"] = max_endpoint_error

    reference = ROOT / protocol["native_reference"]
    previous = json.loads((reference / "summary.json").read_text())
    previous_rows = read_jsonl(reference / "physical_steps.jsonl")
    current_rows = read_jsonl(native_out / "physical_steps.jsonl")
    equal_length = len(previous_rows) == len(current_rows) == 119
    control_error = max((float(np.max(np.abs(np.asarray(old["actual_robot_control"]["values"]) -
                                            np.asarray(new["actual_robot_control"]["values"]))))
                         for old, new in zip(previous_rows, current_rows)), default=0.0)
    nominal_error = max((float(np.max(np.abs(np.asarray(old["nominal_actor_action"]) -
                                            np.asarray(new["nominal_actor_action"]))))
                         for old, new in zip(previous_rows, current_rows)), default=0.0)
    matching_before = sum(old["before"] == new["before"] for old, new in zip(previous_rows, current_rows))
    matching_after = sum(old["after"] == new["after"] for old, new in zip(previous_rows, current_rows))
    matching_complete_snapshots = sum(old["before"] == new["before"] and old["after"] == new["after"]
                                      for old, new in zip(previous_rows, current_rows))
    snapshot_match = equal_length and matching_complete_snapshots == 119
    return_error = abs(previous["outcome"]["return"] - native_summary["outcome"]["return"])
    reproduction = snapshot_match and native_summary["outcome"]["success"] and control_error <= 1e-6 and nominal_error <= 1e-6 and return_error <= 1e-6
    summary["native_reproduction"] = {"passed": reproduction, "same_119_steps": equal_length,
                                      "max_control_difference": control_error, "max_nominal_action_difference": nominal_error,
                                      "matching_before_snapshots": matching_before, "matching_after_snapshots": matching_after,
                                      "matching_complete_transition_snapshots": matching_complete_snapshots,
                                      "all_119_transition_snapshots_exact": snapshot_match,
                                      "cumulative_return_difference": return_error, "reference": str(reference)}
    summary["paired_consequences"] = {
        "states": len(native_rows), "independent_episodes": 1,
        "native_minimum_swept_clearance": min(row["native_swept_clearance"] for row in native_rows),
        "shadow_minimum_swept_clearance": min(row["shadow_swept_clearance"] for row in native_rows),
        "braking_minimum_swept_clearance": min(row["braking_swept_clearance"] for row in native_rows),
        "new_swept_overlap_steps": [row["physical_index_before"] for row in native_rows if row["new_swept_overlap"]],
        "clearance_reduction_steps": [row["physical_index_before"] for row in native_rows if row["shadow_swept_clearance"] < row["native_swept_clearance"]],
        "worst_same_state_clearance_change": min(row["clearance_difference_shadow_minus_native"] for row in native_rows),
        "minimum_expansion": min(row["radii"]["min_delta"] for row in action_rows),
        "maximum_expansion": max(row["radii"]["max_delta"] for row in action_rows),
        "changed_radius_entries": sum(row["radii"]["changed_count"] for row in action_rows),
        "total_radius_entries": sum(row["radii"]["total_count"] for row in action_rows),
        "eligible_radius_rows": sum(row["radii"]["eligible_row_count"] for row in action_rows),
        "shadow_whole_task_success": None, "shadow_whole_task_return": None}
    if not reproduction:
        summary["status"] = "invalid_native_reproduction_no_mechanism_conclusion"
    elif summary["paired_consequences"]["new_swept_overlap_steps"]:
        summary["status"] = "local_counterfactual_physical_overlap_signal_no_closed_loop_harm"
    else:
        summary["status"] = "fixed_pilot_no_decisive_physical_safety_signal_stop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gensafenav_uncertainty_expansion_20261016.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/gensafenav_uncertainty_expansion_20261016")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    assert protocol["fixed_run"] == {
        "case": 0, "seed": 42, "device": "cpu", "wall_seconds": 120, "max_physical_episodes": 1,
        "max_physical_steps": 202, "max_shadow_actor_calls": 202, "delta_meters": 0.1,
        "radius_upper_clip_meters": 1.0, "horizons": [1, 2, 3, 4, 5]}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "launched_script.py")
    shutil.copy2(args.config.resolve(), out / "frozen_protocol.json")
    shutil.copy2(ROOT / protocol["native_runner"].split(":")[0], out / "reused_native_runner.py")
    started = time.monotonic()
    summary = {"status": "starting", "_started": started, "scope": protocol["scope"],
               "extra_activity": {"shadow_actor_calls": 0, "shadow_actor_calls_completed": 0,
                                  "shadow_environment_steps": 0, "model_api_calls": 0, "neural_training_updates": 0},
               "limitations": protocol["limitations"]}

    def timeout(_signum, _frame):
        raise TimeoutError("Frozen 120-second CPU pilot budget exceeded; no automatic retry")

    prior_signal = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120.0)
    code = 0
    try:
        run(args, out, summary, protocol)
    except Exception as error:
        code = 1
        summary.update(status="technical_failure_retained", error=repr(error), traceback=traceback.format_exc())
        print(summary["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_signal)
        summary.pop("_started", None)
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out / "summary.json")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
