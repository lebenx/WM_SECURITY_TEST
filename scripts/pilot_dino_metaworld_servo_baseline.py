#!/usr/bin/env python3
"""Registered, CPU-only goal-proprio servo comparator for the fixed reach panel.

Reuses the frozen environment setup/trace helper, never its model loader. This
is a normal controller comparison on known development cases, not a world-model
causal ablation. A separately invoked --run is required to create environments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/pilot_dino_metaworld_clean_gate.py"
SOURCE_CONFIG = ROOT / "configs/dino_metaworld_mujoco330_clean_v1.json"
SOURCE_SUMMARY = ROOT / "runs/dino_metaworld_mujoco330_clean_v1/summary.json"
SOURCE_ANALYSIS = ROOT / "runs/dino_metaworld_mujoco330_analysis_v1/analysis.json"
DEFAULT_CONFIG = ROOT / "configs/dino_metaworld_servo_baseline_v1.json"
DEFAULT_OUT = ROOT / "runs/dino_metaworld_servo_baseline_v1"
MUJOCO_TARGET = ROOT / ".deps/dino_mujoco330"
OFFICIAL_POLICY = ROOT / ".venv-a2world/lib/python3.11/site-packages/metaworld/policies/sawyer_reach_v3_policy.py"
SEEDS = [1, 2, 3]
BOUNDARIES = [1, 16, 31, 46, 61, 76, 91]
HOLDS = [15] * 6 + [9]
PROTOCOL = {
    "task": "mw-reach", "seeds": SEEDS, "total_environment_steps": 100,
    "warmup_zero_controls": 1, "policy_controls": 99,
    "query_elapsed_steps": BOUNDARIES, "hold_control_counts": HOLDS,
    "policy_inputs": "Only raw float32 current and goal proprio vectors of shape (4,); xyz used, fourth entry ignored",
    "policy": "a_xyz=clip(float32(5)*(goal_xyz-current_xyz),-1,1); a_gripper=0",
    "gain": 5.0, "gripper": 0.0, "action_dtype": "float32",
    "normalization": "None; raw environment actions, never action denormalization",
    "feedback": "Read new proprio only at the seven registered query boundaries; latch the same action within each block",
    "goal": "Unmodified native separate-expert final goal proprio already supplied to the Clean planner",
    "privileged_policy_inputs": False,
    "evaluator_only_fields": ["task target", "TCP", "full state", "expert actions", "reward", "success"],
    "success": "Final native info.success at environment step100, checked against TCP-target distance<=0.05m",
    "early_success_termination": False,
    "setup": "Reuse frozen setup_episode; require every recorded setup array and initial/goal PNG hash exact to the corresponding Clean episode",
    "renderer": "CPU OSMesa", "mujoco": "3.3.0",
    "model_loading": False, "model_calls": 0, "api_requests": 0, "training_updates": 0,
    "attempts_per_seed": 1, "exception_rule": "Stop and preserve partial/error evidence; no replacement, retry, gain tuning or budget change",
}
COMPARISON = {
    "complete_panel_required": True,
    "primary": "Paired final task success on all three fixed seeds",
    "secondary": ["final actual TCP-target distance", "actual reward sum over 99 controls"],
    "servo_covers_clean": "If every Clean-success seed also succeeds under servo and servo success count is no smaller, this panel supplies no observed additional normal-task success benefit for the learned WM planner over this comparator",
    "clean_wins": "A Clean advantage is an advantage over this particular fixed controller, not isolated causal evidence for world forecasting",
    "mixed": "Preserve discordant seeds and describe them without declaring a general ranking",
    "incomplete": "No panel comparison verdict",
    "scope": "Known three-seed development comparison; no population CI/significance, blind confirmation, universal WM-necessity claim, attack or defense claim",
    "old_results": "All previous successful/failed resource gates and scientific conclusions remain unchanged",
}


def load_helper():
    spec = importlib.util.spec_from_file_location("dino_servo_frozen_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def servo_action(current_proprio, goal_proprio):
    """Pure controller: no environment, model, reward or target-state argument."""
    import numpy as np
    for name, value in (("current", current_proprio), ("goal", goal_proprio)):
        if not isinstance(value, np.ndarray) or value.shape != (4,) or value.dtype != np.float32:
            raise ValueError(name + " must be a raw float32 proprio vector of shape (4,)")
        if not np.isfinite(value).all():
            raise ValueError(name + " proprio must be finite")
    action = np.zeros(4, dtype=np.float32)
    action[:3] = np.clip(np.float32(5) * (goal_proprio[:3] - current_proprio[:3]), -1, 1)
    return action


def verify_bindings(helper, value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            helper.require(helper.bind(value["path"]) == {"path": value["path"], "sha256": value["sha256"]},
                           "File binding changed: " + value["path"])
        for child in value.values():
            verify_bindings(helper, child)
    elif isinstance(value, list):
        for child in value:
            verify_bindings(helper, child)


def source_panel(helper):
    source, cpu, _ = helper.preflight(SOURCE_CONFIG)
    clean, _ = helper.read(SOURCE_SUMMARY)
    analysis, _ = helper.read(SOURCE_ANALYSIS)
    helper.require(clean["status"] == "completed" and analysis["status"] == "completed_independent_trace_audit",
                   "Reference panel is incomplete")
    helper.require(clean["config"] == helper.bind(SOURCE_CONFIG)
                   and analysis["source"] == helper.bind(SOURCE_SUMMARY), "Reference linkage differs")
    helper.require([e["seed"] for e in clean["episodes"]] == SEEDS
                   and [e["seed"] for e in analysis["episodes"]] == SEEDS, "Reference seed order differs")
    helper.require(analysis["denominator"] == 3 and analysis["gate_passed"] is True, "Reference readiness not passed")
    verify_bindings(helper, source)
    verify_bindings(helper, clean)
    verify_bindings(helper, analysis)
    for episode, result in zip(clean["episodes"], analysis["episodes"]):
        helper.require(episode["status"] == "completed"
                       and isinstance(episode["outcome"]["final_success"], bool)
                       and episode["outcome"]["final_success"] == result["clean_final_success"],
                       "Reference terminal outcome differs")
    return source, clean, analysis


def prepare(helper, config_path):
    helper.require(not config_path.exists(), "Registration already exists")
    helper.require(not DEFAULT_OUT.exists(), "Run output already exists")
    source, clean, _ = source_panel(helper)
    sources = dict(source["source_sha256"])
    for path in (Path(__file__).resolve(), OFFICIAL_POLICY, OFFICIAL_POLICY.with_name("policy.py"),
                 ROOT / "scripts/analyze_dino_metaworld_clean_gate.py"):
        sources[str(path)] = helper.sha(path)
    config = {
        "schema_version": 1, "status": "registered_before_servo_environment_rollouts",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": helper.bind(__file__), "frozen_helper": helper.bind(HELPER),
        "out_dir": str(DEFAULT_OUT), "protocol": PROTOCOL, "comparison": COMPARISON,
        "source_config": helper.bind(SOURCE_CONFIG), "source_summary": helper.bind(SOURCE_SUMMARY),
        "source_analysis": helper.bind(SOURCE_ANALYSIS), "source_sha256": sources,
        "versions": helper.versions(),
        "official_gain_source": helper.bind(OFFICIAL_POLICY),
        "additional_environment_binaries": source["additional_environment_binaries"],
        "installation_and_load_provenance": source["compatibility_amendment"]["installation_and_load_provenance"],
        "runtime_independent_audit": helper.bind(ROOT / "reports/dino_metaworld_mujoco330_runtime_independent_audit_20261015.json"),
        "reference_artifact_sha256": {str(p.resolve()): helper.sha(p) for p in sorted(SOURCE_SUMMARY.parent.rglob("*")) if p.is_file()},
        "environment": {"CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": str(MUJOCO_TARGET)},
        "known_clean_successes": [e["outcome"]["final_success"] for e in clean["episodes"]],
        "limitations": ["Development comparison registered after the three Clean outcomes were known.",
                        "The servo uses the same goal proprio, not the environment's hidden task target or expert action sequence.",
                        "Feedback opportunities and physical control budget are matched; algorithms, compute and action sequences differ.",
                        "No pure world-forecast causal attribution. A proprio-only CEM would still use predicted proprio and would not be WM-free.",
                        *source["compatibility_amendment"]["limitations"].split("\n")],
        "run_command": "PYTHONPATH=" + str(MUJOCO_TARGET) + " .venv-jepa-native/bin/python scripts/pilot_dino_metaworld_servo_baseline.py --run --config " + str(config_path),
    }
    helper.write(config_path, config, exclusive=True)
    return helper.bind(config_path)


def preflight(helper, config_path):
    import torch
    config, digest = helper.read(config_path)
    helper.require(config["runner"] == helper.bind(__file__) and config["frozen_helper"] == helper.bind(HELPER), "Runner/helper changed")
    helper.require(config["protocol"] == PROTOCOL and config["comparison"] == COMPARISON, "Protocol changed")
    helper.require(config["versions"] == helper.versions(), "Installed runtime changed")
    helper.require(config["source_config"] == helper.bind(SOURCE_CONFIG)
                   and config["source_summary"] == helper.bind(SOURCE_SUMMARY)
                   and config["source_analysis"] == helper.bind(SOURCE_ANALYSIS), "Clean references changed")
    verify_bindings(helper, config)
    for key in ("source_sha256", "reference_artifact_sha256"):
        for name, expected in config[key].items():
            helper.require(helper.sha(name) == expected, "Registered file changed: " + name)
    source_panel(helper)
    helper.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
    helper.require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Renderer changed")
    helper.require(str(MUJOCO_TARGET) in os.environ.get("PYTHONPATH", "").split(os.pathsep), "Isolated MuJoCo path missing")
    return config, {"status": "passed_cpu_preflight", "config": {"path": str(config_path.resolve()), "sha256": digest},
                    "source_files": len(config["source_sha256"]), "reference_artifacts": len(config["reference_artifact_sha256"]),
                    "model_calls": 0, "environment_steps": 0, "cuda_initialized": False}


def audit_decisions(helper, queries, rows, initial_state, goal_proprio):
    import numpy as np
    helper.require([q["elapsed_before"] for q in queries] == BOUNDARIES, "Query schedule differs")
    helper.require([q["executed_control_count"] for q in queries] == HOLDS and len(rows) == 99, "Control budget differs")
    offset = 0
    for index, (query, count) in enumerate(zip(queries, HOLDS)):
        state = initial_state if offset == 0 else rows[offset - 1]["state"]
        current = np.asarray(state[:4], dtype=np.float32)
        goal = np.asarray(goal_proprio, dtype=np.float32)
        action = servo_action(current, goal)
        helper.require(query["current_proprio"] == current.tolist() and query["goal_proprio"] == goal.tolist()
                       and query["action"] == action.tolist(), "Recorded controller input/action differs from actual preceding observation")
        for j, row in enumerate(rows[offset:offset + count]):
            helper.require(row["action"] == action.tolist() and row["planning_call"] == index + 1
                           and row["action_in_chunk"] == j, "Latched action differs")
        offset += count
    return {"query_inputs_verified_from_setup_and_preceding_trace_state": True,
            "all_99_actions_exact_to_registered_servo": True, "decision_count": 7}


def run(helper, config_path):
    import numpy as np
    import torch
    config, validation = preflight(helper, config_path)
    _, clean, _ = source_panel(helper)
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = helper.cfg_for(out, "cpu")
    summary = {"status": "running", "condition": "goal_proprio_servo", "config": validation["config"],
               "runner": helper.bind(__file__), "frozen_helper": helper.bind(HELPER),
               "source_summary": config["source_summary"], "protocol": PROTOCOL,
               "episodes": [], "comparison": None, "model_calls": 0, "model_loads": 0,
               "api_requests": 0, "training_updates": 0, "cuda_initialized": False}
    helper.write(out / "summary.json", summary)
    try:
        with patch.object(helper, "load_model", side_effect=RuntimeError("Model loading forbidden in servo comparator")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA initialization forbidden in CPU comparator")):
            for seed, reference in zip(SEEDS, clean["episodes"]):
                directory = out / f"seed{seed}_servo"
                episode = {"seed": seed, "status": "preparing", "decisions": []}
                summary["episodes"].append(episode)
                env = None
                try:
                    env, initial, goal, setup = helper.setup_episode(cfg, seed, directory)
                    episode["setup"] = setup
                    checks = {"all_setup_arrays_exact": setup["arrays"] == reference["setup"]["arrays"],
                              **{name + "_bytes_exact": setup[name]["sha256"] == reference["setup"][name]["sha256"] for name in ("initial_image", "goal_image")}}
                    episode["clean_setup_parity"] = {"checks": checks, "all_exact": all(checks.values())}
                    helper.require(all(checks.values()), "Servo setup differs from registered Clean")
                    current = initial["proprio"][-1].detach().cpu().numpy().copy()
                    target = goal["proprio"][-1].detach().cpu().numpy().copy()
                    rows = []
                    with (directory / "trace.jsonl").open("x") as handle:
                        for query_id, (elapsed, count) in enumerate(zip(BOUNDARIES, HOLDS), 1):
                            helper.require(env.elapsed_steps() == elapsed, "Unexpected query boundary")
                            action = servo_action(current, target)
                            query = {"index": query_id, "elapsed_before": elapsed, "current_proprio": current.tolist(),
                                     "goal_proprio": target.tolist(), "action": action.tolist(), "executed_control_count": 0}
                            episode["decisions"].append(query)
                            for local_index in range(count):
                                _, info, row = helper.trace_step(env, torch.from_numpy(action.copy()), query_id, local_index)
                                rows.append(row); handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                                query["executed_control_count"] += 1
                                helper.require(row["done"] == (len(rows) == 99), "Unexpected termination")
                            # The controller consumes feedback only at the next query boundary.
                            current = info["proprio"][-1].detach().cpu().numpy().copy()
                            helper.write(directory / "decisions.json", episode["decisions"])
                            helper.write(out / "summary.json", summary)
                    with np.load(setup["setup"]["path"], allow_pickle=False) as arrays:
                        episode["decision_audit"] = audit_decisions(helper, episode["decisions"], rows, arrays["initial_state"], target)
                    episode.update(status="completed", outcome=helper.outcome(rows, directory),
                                   queries_file=helper.bind(directory / "decisions.json"))
                    print(json.dumps({"seed": seed, "status": "completed", "final_success": episode["outcome"]["final_success"]}), flush=True)
                except BaseException as error:
                    episode.update(status="error", exception_type=type(error).__name__, exception=str(error))
                    raise
                finally:
                    if env is not None: env.close()
                    helper.write(out / "summary.json", summary)
        helper.require(len(summary["episodes"]) == 3 and not torch.cuda.is_initialized(), "Incomplete panel or CUDA initialized")
        paired = [{"seed": a["seed"], "clean_final_success": a["outcome"]["final_success"],
                   "servo_final_success": b["outcome"]["final_success"],
                   "clean_final_distance": a["outcome"]["final_reach_distance"], "servo_final_distance": b["outcome"]["final_reach_distance"],
                   "clean_return": a["outcome"]["reward_sum"], "servo_return": b["outcome"]["reward_sum"]}
                  for a, b in zip(clean["episodes"], summary["episodes"])]
        covers = all(not row["clean_final_success"] or row["servo_final_success"] for row in paired)
        clean_count = sum(row["clean_final_success"] for row in paired)
        servo_count = sum(row["servo_final_success"] for row in paired)
        summary.update(status="completed", comparison={"definition": COMPARISON, "paired_results": paired,
                       "denominator": 3, "clean_successes": clean_count, "servo_successes": servo_count,
                       "servo_covers_clean_successes": covers,
                       "no_observed_additional_wm_task_success_benefit_over_servo": covers and servo_count >= clean_count,
                       "forecast_causal_benefit_established": False})
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary["cuda_initialized"] = torch.cuda.is_initialized()
        helper.write(out / "summary.json", summary)


def self_test():
    import numpy as np
    checks = []
    current = np.asarray([0, 0, 0, .2], dtype=np.float32)
    goal = np.asarray([.1, -.3, .4, .9], dtype=np.float32)
    assert np.array_equal(servo_action(current, goal), np.asarray([.5, -1, 1, 0], dtype=np.float32))
    checks.append("official_gain_clipping_gripper_and_dtype")
    other = goal.copy(); other[3] = -99
    assert np.array_equal(servo_action(current, other), servo_action(current, goal))
    checks.append("goal_gripper_not_used")
    for bad in (np.zeros(39, dtype=np.float32), np.zeros(4, dtype=np.float64), np.asarray([0, 0, np.nan, 0], dtype=np.float32)):
        try: servo_action(bad, goal)
        except ValueError: pass
        else: raise AssertionError("Invalid policy input accepted")
    checks.append("reject_fullstate_wrong_dtype_and_nonfinite")
    assert sum(HOLDS) == 99 and [1 + sum(HOLDS[:i]) for i in range(7)] == BOUNDARIES
    checks.append("fixed_seven_feedback_boundaries_and_budget")
    return {"status": "passed", "checks": checks, "environment_steps": 0, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "preflight", "self-test", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    helper = load_helper()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    helper.bootstrap(cpu=True)
    if args.self_test:
        result = self_test()
    elif args.prepare:
        result = prepare(helper, args.config)
    elif args.preflight:
        _, result = preflight(helper, args.config)
    else:
        run(helper, args.config)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
