#!/usr/bin/env python3
"""Conditional native JEPA-WM reach-wall execution after the fixed CPU gate.

Preparation and preflight are read-only CPU checks. --run is the only entry
that creates a model/environment; it uses native full-batch FP32 CEM unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CPU_RUNNER = ROOT / "scripts/pilot_jepa_metaworld_reach_wall.py"
CPU_CONFIG = ROOT / "configs/jepa_metaworld_reach_wall_cpu_v1.json"
CPU_SUMMARY = ROOT / "runs/jepa_metaworld_reach_wall_cpu_v1/summary.json"
TRACE_AUDITOR = ROOT / "scripts/analyze_dino_metaworld_clean_gate.py"
DEFAULT_CONFIG = ROOT / "configs/jepa_metaworld_reach_wall_native_v1.json"
DEFAULT_OUT = ROOT / "runs/jepa_metaworld_reach_wall_native_v1"
ALLOCATOR = "expandable_segments:True"


def load_modules():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reach_wall_cpu_adapter", CPU_RUNNER)
    wall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wall)
    helper = wall.load_module("reach_wall_native_frozen_helper", wall.HELPER)
    servo = wall.load_module("reach_wall_native_frozen_servo", wall.SERVO_HELPER)
    audit = wall.load_module("reach_wall_native_trace_auditor", TRACE_AUDITOR)
    return wall, helper, servo, audit


def verify_refs(helper, value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            helper.require(helper.bind(value["path"]) == {"path": value["path"], "sha256": value["sha256"]}, "Artifact binding changed: " + value["path"])
        for child in value.values():
            verify_refs(helper, child)
    elif isinstance(value, list):
        for child in value:
            verify_refs(helper, child)


def cpu_evidence(wall, helper, servo, audit):
    """Rebuild the prerequisites from fixed artifacts, never a partial status."""
    import numpy as np
    config, config_sha = helper.read(CPU_CONFIG)
    cpu, summary_sha = helper.read(CPU_SUMMARY)
    helper.require(config["runner"] == helper.bind(CPU_RUNNER) and config["protocol"] == wall.PROTOCOL
                   and config["gate"] == wall.GATE, "CPU registration differs")
    helper.require(cpu["status"] == "completed_cpu_validation" and cpu["config"] == {"path": str(CPU_CONFIG), "sha256": config_sha}
                   and cpu["runner"] == config["runner"] and cpu["protocol"] == wall.PROTOCOL and cpu["gate"] == wall.GATE,
                   "CPU panel is incomplete or unbound")
    helper.require(config["versions"] == helper.versions() and config["source_sha256"] == wall.inventory(helper), "CPU sources/runtime changed")
    helper.require(cpu["planned_branches"] == config["planned_branches"] ==
                   [{"seed": s, "condition": b} for s in wall.SEEDS for b in wall.BRANCHES], "CPU branch schedule differs")
    verify_refs(helper, config)
    verify_refs(helper, cpu)
    helper.require(cpu["model_forward_calls"] == cpu["api_requests"] == cpu["training_updates"] == 0
                   and cpu["cuda_initialized"] is False, "CPU execution scope differs")
    loading = cpu["strict_model_load"]
    helper.require(loading and loading["strict_loads"] and loading["all_parameters_float32"]
                   and loading["model_forward_calls"] == 0 and loading["model_action_dimension"] == 20
                   and all(not c["missing"] and not c["unexpected"] for c in loading["strict_loads"]), "CPU strict loading failed")
    helper.require([e["seed"] for e in cpu["episodes"]] == wall.SEEDS, "CPU fixed seeds incomplete")
    controls = 0
    for episode in cpu["episodes"]:
        helper.require(episode["status"] == "completed" and episode["prerequisites_passed"] is True
                       and [b["condition"] for b in episode["branches"]] == wall.BRANCHES, "CPU episode incomplete")
        reference = episode["branches"][0]
        for branch in episode["branches"]:
            helper.require(branch["status"] == "completed", "CPU branch incomplete")
            setup = branch["setup"]
            helper.require(setup["seed"] == episode["seed"] and setup["elapsed_after_warmup"] == 1
                           and not setup["initial_success"] and setup["expert_final_success"], "CPU setup prerequisites failed")
            arrays = audit.setup_arrays(setup)
            helper.require(all(wall.setup_checks(reference["setup"], setup).values()), "CPU branch setups differ")
            _, rows = audit.trace_outcome(branch["outcome"], setup, arrays, branch["condition"].startswith("noop"))
            controls += len(rows)
            final = branch["final_observation"]
            with np.load(audit.bound(final["npz"]), allow_pickle=False) as saved:
                helper.require({k: helper.array_digest(saved[k]) for k in saved.files} == final["arrays"], "CPU final array digest differs")
            if not branch["condition"].startswith("noop"):
                queries, _ = helper.read(audit.bound(branch["decisions"]))
                helper.require([q["elapsed_before"] for q in queries] == wall.BOUNDARIES
                               and [q["executed_control_count"] for q in queries] == wall.HOLDS, "CPU servo schedule differs")
                target = arrays["goal_proprio"][-1].astype(np.float32)
                offset = 0
                for index, (q, count) in enumerate(zip(queries, wall.HOLDS), 1):
                    current = np.asarray((arrays["initial_state"] if offset == 0 else rows[offset - 1]["state"])[:4], dtype=np.float32)
                    action = (servo.servo_action(current, target) if branch["condition"] == "straight_servo"
                              else wall.wall_aware_action(servo, current, target)).tolist()
                    helper.require(q["index"] == index and q["current_proprio"] == current.tolist()
                                   and q["goal_proprio"] == target.tolist() and q["action"] == action, "CPU servo inputs differ")
                    helper.require(all(r["action"] == action and r["planning_call"] == index and r["action_in_chunk"] == j
                                       for j, r in enumerate(rows[offset:offset + count])), "CPU servo execution differs")
                    offset += count
        first, second = episode["branches"][:2]
        helper.require(first["outcome"]["trace"]["sha256"] == second["outcome"]["trace"]["sha256"]
                       and first["final_observation"]["arrays"] == second["final_observation"]["arrays"]
                       and first["final_observation"]["image"]["sha256"] == second["final_observation"]["image"]["sha256"], "CPU noop repeat differs")
    rebuilt = wall.decision(cpu["episodes"], True)
    helper.require(cpu["decision"] == rebuilt and rebuilt["gpu_progression_allowed"] is True,
                   "CPU prerequisites or mathematical screen prohibit GPU registration/execution")
    return config, cpu, {"cpu_config": {"path": str(CPU_CONFIG), "sha256": config_sha},
                         "cpu_summary": {"path": str(CPU_SUMMARY), "sha256": summary_sha},
                         "verified_control_count": controls, "decision": rebuilt}


def runtime(wall, helper):
    import mujoco
    helper.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == ALLOCATOR, "Allocator revision differs")
    helper.require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Renderer differs")
    helper.require(str(wall.MUJOCO_TARGET) in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                   and Path(mujoco.__file__).resolve().is_relative_to(wall.MUJOCO_TARGET)
                   and mujoco.__version__ == mujoco.mj_versionString() == "3.3.0", "MuJoCo runtime differs")
    return {"mujoco_import": str(Path(mujoco.__file__).resolve()), "mujoco_engine": mujoco.mj_versionString(),
            "PYTORCH_CUDA_ALLOC_CONF": ALLOCATOR, "renderer": "CPU OSMesa"}


def native_protocol(wall):
    # Keep the CPU-only zero-forward declaration out of the GPU protocol.
    protocol = {k: v for k, v in wall.PROTOCOL["future_native_gpu"].items() if k != "execution"}
    protocol.update(task="mw-reach-wall", seeds=wall.SEEDS, condition="clean_native", attempts_per_seed=1,
                    total_environment_steps=100, warmup_zero_controls=1, policy_control_budget=99,
                    query_elapsed_steps=wall.BOUNDARIES, executed_control_counts=wall.HOLDS,
                    goal=wall.PROTOCOL["goal"], primary_success=wall.PROTOCOL["primary_success"],
                    physical_seed_rule=wall.PROTOCOL["physical_seed_rule"],
                    renderer="CPU OSMesa", mujoco="3.3.0 isolated target", threads=4,
                    api_requests=0, training_updates=0)
    return protocol


def prepare(wall, helper, servo, audit, config_path, out):
    helper.require(not config_path.exists() and not out.exists(), "GPU registration/output already exists")
    cpu_config, cpu, evidence = cpu_evidence(wall, helper, servo, audit)
    source = {**cpu_config["source_sha256"], str(Path(__file__).resolve()): helper.sha(__file__),
              str(TRACE_AUDITOR): helper.sha(TRACE_AUDITOR)}
    config = {
        "schema_version": 1, "status": "registered_before_any_reach_wall_native_model_rollout",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": helper.bind(__file__), "cpu_runner": helper.bind(CPU_RUNNER),
        "frozen_native_helper": helper.bind(wall.HELPER), "trace_auditor": helper.bind(TRACE_AUDITOR),
        "cpu_config": evidence["cpu_config"], "cpu_summary": evidence["cpu_summary"], "cpu_evidence_audit": evidence,
        "out_dir": str(out.resolve()), "seeds": wall.SEEDS,
        "protocol": native_protocol(wall), "cpu_protocol": wall.PROTOCOL, "gate": wall.GATE, "model": helper.bind(wall.MODEL),
        "dino_encoder": helper.bind(helper.DINO_WEIGHTS), "official_config": helper.bind(wall.OFFICIAL),
        "source_sha256": source, "versions": helper.versions(), "runtime": runtime(wall, helper),
        "cpu_artifact_sha256": {str(p.resolve()): helper.sha(p) for p in sorted(CPU_SUMMARY.parent.rglob("*")) if p.is_file()},
        "execution": {"native_runs_per_seed": 1, "seed_order": wall.SEEDS,
                      "native_gc_agent_and_cem": "Unmodified vendor GC_Agent/CEM; one agent and its local CUDA generator across all3 fixed episodes",
                      "loop": "Same native loop as frozen DINO helper.run; independent adapter uses JEPA cfg/checkpoint and checks all4 CPU setup branches",
                      "precision": "FP32, full300 candidate batch; no autocast/microbatch",
                      "allocator": ALLOCATOR, "cudnn_benchmark": True,
                      "failure_rule": "Terminal on any runtime/parity failure, preserve all partial data; no retry/replacement and no native task-failure verdict for incomplete runs",
                      "cpu_comparators": "Only frozen CPU results; not rerun during GPU stage"},
        "scope": "Clean normal capability screen, not attack, defense, or pure world-model causal attribution. Passing against wall-aware+noop does not establish advantage over every ordinary controller; straight-P5 remains descriptive.",
        "adaptations": cpu_config["adaptations"], "limitations": cpu_config["limitations"],
        "run_command": "PYTHONPATH=" + str(wall.MUJOCO_TARGET) + " PYTORCH_CUDA_ALLOC_CONF=" + ALLOCATOR
                       + " .venv-jepa-native/bin/python scripts/pilot_jepa_metaworld_reach_wall_native.py --run --config " + str(config_path.resolve()),
    }
    helper.write(config_path, config, exclusive=True)
    return helper.bind(config_path)


def preflight(wall, helper, servo, audit, path):
    config, digest = helper.read(path)
    helper.require(config["runner"] == helper.bind(__file__) and config["cpu_runner"] == helper.bind(CPU_RUNNER)
                   and config["protocol"] == native_protocol(wall) and config["cpu_protocol"] == wall.PROTOCOL
                   and config["gate"] == wall.GATE and config["seeds"] == wall.SEEDS,
                   "Native registration differs")
    helper.require(config["versions"] == helper.versions() and config["runtime"] == runtime(wall, helper), "Runtime changed")
    verify_refs(helper, config)
    for field in ("source_sha256", "cpu_artifact_sha256"):
        for name, digest_expected in config[field].items():
            helper.require(helper.sha(name) == digest_expected, "Registered source/evidence changed: " + name)
    _, cpu, evidence = cpu_evidence(wall, helper, servo, audit)
    helper.require(config["cpu_evidence_audit"] == evidence, "CPU audit binding differs")
    return config, cpu, {"status": "passed_cpu_preflight", "config": {"path": str(path.resolve()), "sha256": digest},
                         "model_instances_created": 0, "model_forward_calls": 0, "environment_steps": 0,
                         "source_files": len(config["source_sha256"]), "cpu_artifacts": len(config["cpu_artifact_sha256"])}


def evaluate_gate(wall, episodes, references):
    if len(episodes) != 3 or any(e["status"] != "completed" for e in episodes):
        return None
    count = sum(e["outcome"]["final_success"] for e in episodes)
    comparators = {b: sum(e["branches"][wall.BRANCHES.index(b)]["outcome"]["final_success"] for e in references)
                   for b in ("noop_repeat0", "straight_servo", "wall_aware_servo")}
    return {"passed": count >= 2 and count - comparators["wall_aware_servo"] >= 2 and count - comparators["noop_repeat0"] >= 2,
            "native_final_successes": count, "cpu_final_successes": comparators,
            "native_minus_comparator": {b: count - n for b, n in comparators.items()}, "definition": wall.GATE,
            "interpretation": "Three fixed development examples; no significance or universal superiority claim"}


def run(wall, helper, servo, audit, config_path):
    import numpy as np
    import torch
    from PIL import Image
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    config, cpu, validation = preflight(wall, helper, servo, audit, config_path)
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    cfg = wall.cfg_for(helper, out, "cuda")
    summary = {"status": "loading", "config": validation["config"], "runner": helper.bind(__file__),
               "cpu_summary": config["cpu_summary"], "protocol": config["protocol"], "episodes": [], "gate": None,
               "api_requests": 0, "training_updates": 0,
               "cuda_runtime": {"torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}}
    helper.write(out / "summary.json", summary)
    try:
        helper.require(torch.cuda.is_available(), "CUDA required; no fallback model backend")
        summary["cuda_memory_before_load"] = {"free": torch.cuda.mem_get_info()[0], "total": torch.cuda.mem_get_info()[1]}
        with patch.object(helper, "MODEL", wall.MODEL):
            model, pre, loading = helper.load_model(cfg, "cuda")
        summary["model_load"] = loading
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
        for seed, reference in zip(wall.SEEDS, cpu["episodes"]):
            helper.require(reference["seed"] == seed, "Reference seed order changed")
            directory = out / f"seed{seed}_clean"
            episode = {"seed": seed, "status": "preparing", "queries": [], "pending_query": None}
            summary["episodes"].append(episode)
            summary["status"] = "running"
            helper.write(out / "summary.json", summary)
            env = None
            try:
                env, obs, goal, setup = helper.setup_episode(cfg, seed, directory)
                episode["setup"] = setup
                checks = {b["condition"]: wall.setup_checks(setup, b["setup"]) for b in reference["branches"]}
                episode["cpu_setup_parity"] = {"checks": checks, "all_exact": all(all(v.values()) for v in checks.values())}
                helper.require(episode["cpu_setup_parity"]["all_exact"], "Native physical/observed/expert setup differs from CPU branches")
                episode["matched_cpu_outcomes"] = {b["condition"]: b["outcome"] for b in reference["branches"]}
                agent.set_goal(goal)
                helper.require(agent.planner.num_samples == 300, "Native candidate batch changed")
                rows = []
                with (directory / "trace.jsonl").open("x") as trace:
                    while env.steps_left() > 0:
                        elapsed = env.elapsed_steps()
                        steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                        episode["pending_query"] = {"index": len(episode["queries"]) + 1, "elapsed_before": elapsed,
                                                    "planning_steps_left": steps_left, "horizon": min(6, steps_left)}
                        helper.write(out / "summary.json", summary)
                        tick = time.monotonic()
                        normalized = agent.act(obs, steps_left=steps_left)
                        actions = pre.denormalize_actions(normalized.cpu().reshape(-1, 4))
                        helper.require(torch.isfinite(actions).all(), "Nonfinite native action")
                        query = {**episode["pending_query"], "normalized_macro_actions": normalized.cpu().tolist(),
                                 "physical_actions": actions.tolist(), "planned_control_count": len(actions), "executed_control_count": 0,
                                 "minimum_cost_per_iteration": agent._prev_losses.cpu().tolist(), "elapsed_seconds": time.monotonic() - tick,
                                 "cuda_allocated_bytes": torch.cuda.memory_allocated(), "cuda_reserved_bytes": torch.cuda.memory_reserved()}
                        episode["queries"].append(query)
                        episode["pending_query"] = None
                        helper.require(len(query["minimum_cost_per_iteration"]) == 15, "Native iteration count changed")
                        for index, action in enumerate(actions):
                            visual, info, row = helper.trace_step(env, action, query["index"], index)
                            rows.append(row); trace.write(json.dumps(row, allow_nan=False) + "\n"); trace.flush()
                            query["executed_control_count"] += 1
                            obs = TensorDict({"visual": visual, "proprio": info["proprio"]})
                            if row["done"]:
                                break
                        helper.write(directory / "queries.json", episode["queries"])
                        helper.write(out / "summary.json", summary)
                        print(json.dumps({"seed": seed, "query": query["index"], "environment_steps": env.elapsed_steps(),
                                          "reach_distance": rows[-1]["reach_distance"], "success": rows[-1]["success"]}), flush=True)
                        if rows[-1]["done"]:
                            break
                helper.require([q["elapsed_before"] for q in episode["queries"]] == wall.BOUNDARIES
                               and [q["executed_control_count"] for q in episode["queries"]] == wall.HOLDS
                               and [q["planned_control_count"] for q in episode["queries"]] == [15] * 6 + [10], "Native control schedule differs")
                outcome = helper.outcome(rows, directory)
                final = {**helper.physical(env), "pixels": visual.detach().cpu().numpy().copy(),
                         "proprio": info["proprio"].detach().cpu().numpy().copy(), "state": np.asarray(info["state"]).copy()}
                np.savez(directory / "final.npz", **final)
                Image.fromarray(final["pixels"][-1].transpose(1, 2, 0)).save(directory / "final.png")
                episode.update(status="completed", outcome=outcome,
                               final_observation={"arrays": {k: helper.array_digest(v) for k, v in final.items()},
                                                  "npz": helper.bind(directory / "final.npz"), "image": helper.bind(directory / "final.png")})
            except BaseException as error:
                episode.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
                raise
            finally:
                if env is not None:
                    env.close()
                helper.write(out / "summary.json", summary)
        summary["gate"] = evaluate_gate(wall, summary["episodes"], cpu["episodes"])
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if torch.cuda.is_initialized():
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        helper.write(out / "summary.json", summary)


def self_test(wall, helper):
    refs = [{"branches": [{"outcome": {"final_success": False}} for _ in wall.BRANCHES]} for _ in wall.SEEDS]
    episodes = [{"status": "completed", "outcome": {"final_success": i < 2}} for i in range(3)]
    helper.require(evaluate_gate(wall, episodes, refs)["passed"], "Two native vs zero baseline must pass")
    refs[0]["branches"][3]["outcome"]["final_success"] = True
    helper.require(not evaluate_gate(wall, episodes, refs)["passed"], "Wall-aware net gain below2 must fail")
    refs[0]["branches"][3]["outcome"]["final_success"] = False
    refs[0]["branches"][0]["outcome"]["final_success"] = True
    helper.require(not evaluate_gate(wall, episodes, refs)["passed"], "Noop net gain below2 must fail")
    helper.require(evaluate_gate(wall, episodes[:2], refs) is None, "Incomplete panel must have no gate")
    episodes[2]["status"] = "error"
    helper.require(evaluate_gate(wall, episodes, refs) is None, "Runtime error must have no gate")
    helper.require([1 + sum(wall.HOLDS[:i]) for i in range(7)] == wall.BOUNDARIES and sum(wall.HOLDS) == 99, "Schedule differs")
    return {"status": "passed", "checks": 6, "runner": helper.bind(__file__), "model_instances_created": 0,
            "model_forward_calls": 0, "environment_steps": 0, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-registration", "preflight", "run", "self-test"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOCATOR)
    wall, helper, servo, audit = load_modules()
    helper.bootstrap(cpu=not args.run)
    if args.run:
        run(wall, helper, servo, audit, args.config.resolve())
        return
    if args.prepare_registration:
        result = prepare(wall, helper, servo, audit, args.config.resolve(), args.out_dir.resolve())
    elif args.preflight:
        result = preflight(wall, helper, servo, audit, args.config.resolve())[2]
    else:
        result = self_test(wall, helper)
    if args.report:
        helper.write(args.report, result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
