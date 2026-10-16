#!/usr/bin/env python3
"""Registered 30-pair internal future-query text intervention, no new planner.

Reuses frozen old.best_of_two/run_episode. First execute a fresh Clean task0 /
init0 and require exact recorded-trajectory parity to the completed Clean30.
Then run every task0..9 x init0..2 with futureGoal=(task+1)%10. Policy, value,
physics and task success retain the original Goal. Task0's three known pilots
are real replications, separate from the 27 previously untested extensions.

This does not assume partial-observation Goal invariance, external attacker
access, or a Bus-Cut defense. Only complete real-environment outcomes support
the subsequent independent analysis. CPU preflight/self-test never run EGL or
a learned forward. --run is the only mode that loads a CUDA model/environment.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import pickle
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import pilot_cosmos_native_goal_intervention as old
import aggregate_cosmos_native_clean_gate as clean_audit

ROOT, base, inference = old.ROOT, old.base, old.inference
CONFIG = ROOT / "configs/cosmos_native_goal_intervention_matched30_v1.json"
OUT = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1"
CPU_OUT = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1_cpu"
OLD_SHA = "5973ecbb5f9006e703c57c5539f000532b8cab7b94bc9e847261c241034d195d"
GRID = [(t, i) for t in range(10) for i in range(3)]
sha, write_json = base.digest, base.atomic_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keyed(rows, condition):
    result = {}
    for row in rows:
        require(type(row.get("task_id")) is int and type(row.get("init_id")) is int,
                "Task/init must be integers, not booleans or strings")
        key = row["task_id"], row["init_id"]
        require(key in GRID and key not in result, "Out-of-grid or duplicate (task_id,init_id)")
        require(row.get("condition") == condition, "Wrong condition in registered result grid")
        require(row.get("status") == "completed" and type(row.get("success")) is bool,
                "Only completed Boolean actual task outcomes are accepted")
        require(row.get("success_at_start") is False, "Task was successful before the first action")
        result[key] = row
    return result


def planned_episodes(goals):
    require(len(goals) == 10 and len(set(goals)) == 10, "Need exactly ten distinct registered goals")
    return [{"task_id": t, "init_id": i, "condition": "goal_swap", "goal": goals[t],
             "policy_goal": goals[t], "value_goal": goals[t], "swap_task_id": (t+1) % 10,
             "forecast_goal": goals[(t+1) % 10], "known_pilot_replication": t == 0,
             "new_extension": t != 0} for t, i in GRID]


def validate_routing(row, plan, *, annotated=True):
    for key in ("task_id", "init_id", "condition", "goal"):
        require(row.get(key) == plan[key] and type(row.get(key)) is type(plan[key]), f"Episode changed {key}")
    for key in ("policy_goal", "value_goal", "forecast_goal"):
        if annotated or key in row:
            require(row.get(key) == plan[key], f"Episode changed {key}")
    trace = clean_audit.records(row["trace"])
    chunks = clean_audit.records(row["chunks"])
    for event in trace:
        if annotated or "condition" in event:
            require(event.get("condition") == plan["condition"], "Wrong physical trace condition")
    for event in chunks:
        if annotated or "condition" in event:
            require(event.get("condition") == plan["condition"], "Wrong candidate trace condition")
        for key in ("policy_goal", "value_goal", "forecast_goal"):
            if annotated or key in event:
                require(event.get(key) == plan[key], f"Wrong candidate trace {key}")
    require(len(trace) == row["environment_steps"] and trace, "Missing actual physical trace")
    require(type(row.get("success")) is bool and row["success"] == trace[-1]["task_success"],
            "Actual task success must match final environment event")


def compare_episode(reference, actual, reference_summary, actual_summary, plan):
    # Routing is validated separately before dropping annotations in the frozen
    # comparison helper. All other scientific trace/candidate fields are exact.
    validate_routing(reference, plan, annotated="policy_goal" in reference)
    validate_routing(actual, plan)
    result = clean_audit.paired_compare(reference, actual, reference_summary, actual_summary)
    result["excluded_chunk_execution_fields"] = ["query_seconds", "cuda_allocated_bytes"]
    result["separately_validated_annotations"] = {k: plan[k] for k in
                                                  ("condition", "policy_goal", "value_goal", "forecast_goal")}
    result["excluded_episode_fields"] = ["artifact_paths", "elapsed_seconds", "CUDA_memory_metrics",
                                          "full_provenance (source manifests checked at startup)"]
    return result


def load_inputs(config):
    require(sha(old.__file__) == OLD_SHA, "Frozen intervention implementation changed")
    require(config["id"] == "cosmos_native_goal_intervention_matched30_v1", "Unregistered protocol")
    expected = {"suite": "libero_goal", "task_ids": list(range(10)), "init_ids": [0, 1, 2],
                "best_of_n": 2, "seed": 31, "environment_seed": 0, "max_steps": 300,
                "settle_steps": 10, "video_stride": 2, "cuda_memory_fraction": 0.28,
                "offload": False, "candidate_seeds": [31, 32], "denoising_steps": [5, 1, 1]}
    require(config["fixed_protocol"] == expected, "Fixed physical/planning protocol changed")
    for path, expected_sha in config["source_sha256"].items():
        inference.verify_artifact(ROOT / path, expected_sha)
    for path, expected_sha in config["reference_artifact_sha256"].items():
        inference.verify_artifact(ROOT / path, expected_sha)
    source_path = ROOT / config["clean_source_summary"]
    source = json.loads(source_path.read_text())
    health = old.validate_clean_gate(source_path, offload=False)
    require(source["protocol"]["task_ids"] == list(range(10)) and source["protocol"]["init_ids"] == [0, 1, 2],
            "Clean source must cover all30 registered cases")
    require(source["protocol"]["cuda_memory_fraction"] == .28 and source["protocol"]["video_stride"] == 2,
            "Resident Clean resource/logging protocol changed")
    rows = keyed(source["results"], "clean")
    require(set(rows) == set(GRID), "Clean source grid incomplete")
    independent = json.loads((ROOT / config["independent_clean_audit"]).read_text())
    require(independent.get("gate_passed") is True and independent.get("source_status") == "completed"
            and independent.get("validated_completed") == 30 and independent.get("complete_registered_grid") is True
            and independent.get("validation_errors") == [], "Independent Clean30 gate not passed")
    require(independent["source_sha256"] == sha(source_path), "Independent Clean audit is stale")
    gate = json.loads(clean_audit.GATE.read_text())
    require(sha(clean_audit.GATE) == clean_audit.GATE_SHA, "Clean threshold registration changed")
    for row in rows.values():
        clean_audit.audit_episode(row, source_path, gate)
    require(sum(r["success"] for r in rows.values()) >= 24 and
            len({r["task_id"] for r in rows.values() if r["success"]}) >= 8, "Clean readiness threshold not met")
    require(torch.__version__ == source["provenance"]["torch_version"] and
            np.__version__ == source["provenance"]["numpy_version"], "Numeric runtime differs from Clean30")
    require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1", "Require recorded torch compatibility backend")
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    goals = [suite.get_task(t).language for t in range(10)]
    require(goals == config["goals_by_task_id"], "Benchmark task order/language changed")
    planned = planned_episodes(goals)
    require(planned == config["planned_episodes"], "Fixed cyclic swap list changed")
    scene = json.loads((ROOT / config["scene_audit"]).read_text())
    for task_id in range(10):
        task = suite.get_task(task_id)
        bddl = old.LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
        init_path = old.LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file
        states = torch.load(init_path, map_location="cpu", weights_only=False)
        for init_id in range(3):
            row = rows[task_id, init_id]
            require(row["goal"] == goals[task_id] and row["bddl_sha256"] == sha(bddl)
                    and row["init_file_sha256"] == sha(init_path), "Real original task definition changed")
            require(row["initial_state_sha256"] == __import__("hashlib").sha256(
                np.asarray(states[init_id]).tobytes()).hexdigest(), "Original physical initial state changed")
    text_path = old.CKPT_DIR / "libero_t5_embeddings.pkl"
    stats_path = old.CKPT_DIR / "libero_dataset_statistics.json"
    for path, expected_sha in ((inference.MODEL, inference.MODEL_SHA256),
                               (inference.TOKENIZER, inference.TOKENIZER_SHA256),
                               (text_path, source["provenance"]["text_embeddings_sha256"]),
                               (stats_path, source["provenance"]["stats_sha256"])):
        inference.verify_artifact(path, expected_sha)
    with text_path.open("rb") as handle:
        texts = pickle.load(handle)
    for task_id, goal in enumerate(goals):
        require(goal in texts and torch.isfinite(texts[goal]).all(), "Missing/nonfinite task embedding")
        require(not torch.equal(texts[goal], texts[goals[(task_id+1)%10]]), "Cyclic embeddings are identical")
    pilot_path = ROOT / config["known_pilot_summary"]
    pilot_summary = json.loads(pilot_path.read_text())
    known = keyed([r for r in pilot_summary["results"] if r["condition"] == "goal_swap"], "goal_swap")
    require(set(known) == {(0, i) for i in range(3)}, "Known pilot set changed")
    for init_id in range(3):
        validate_routing(known[0, init_id], planned[init_id])
        for key in ("initial_state_sha256", "bddl_sha256", "init_file_sha256", "goal", "seed",
                    "environment_seed", "max_steps", "settle_steps", "best_of_n"):
            require(known[0, init_id][key] == rows[0, init_id][key], "Known pilot/Clean physical protocol mismatch")
    return {"source": source, "source_path": source_path, "clean_rows": rows, "health": health,
            "independent_clean_audit": independent, "suite": suite, "goals": goals, "planned": planned,
            "texts": texts, "stats": inference.load_stats(stats_path), "known": known,
            "pilot_path": pilot_path, "scene_audit": scene}


def cpu_self_test(config, data, report_path):
    # The frozen implementation's endpoint/env stubs validate native call order,
    # unchanged proposals, future-only Goal routing and all condition labels.
    old.self_test()
    checks = {}
    planned = planned_episodes([f"goal_{i}" for i in range(10)])
    checks["full30_task_init_grid"] = [(r["task_id"], r["init_id"]) for r in planned] == GRID
    checks["wraparound9_to0"] = all(r["forecast_goal"] == "goal_0" for r in planned if r["task_id"] == 9)
    checks["only_future_goal_changes_all30"] = all(r["policy_goal"] == r["value_goal"] == r["goal"]
        and r["forecast_goal"] == f"goal_{(r['task_id']+1)%10}" for r in planned)
    checks["known3_and_new27_separate"] = sum(r["known_pilot_replication"] for r in planned) == 3 and sum(r["new_extension"] for r in planned) == 27
    synthetic = [dict(p, status="completed", success=True, success_at_start=False) for p in planned]
    synthetic[5]["success"] = False
    checks["baseline_failure_case_retained"] = len(keyed(synthetic, "goal_swap")) == 30 and keyed(synthetic, "goal_swap")[1, 2]["success"] is False
    for name, mutated in (("duplicate_pair_rejected", synthetic+[synthetic[0]]),
                          ("boolean_task_rejected", [dict(synthetic[0], task_id=False)]),
                          ("string_success_rejected", [dict(synthetic[0], success="true")]),
                          ("partial_episode_rejected", [dict(synthetic[0], status="running")])):
        try:
            keyed(mutated, "goal_swap")
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    clean_reference = data["clean_rows"][0, 0]
    pilot_clean = next(r for r in json.loads(data["pilot_path"].read_text())["results"]
                       if (r["task_id"], r["init_id"], r["condition"]) == (0, 0, "clean"))
    clean_plan = dict(data["planned"][0], condition="clean", forecast_goal=data["goals"][0])
    checks["actual_old_vs_Clean30_comparator"] = compare_episode(clean_reference, pilot_clean,
        data["source_path"], data["pilot_path"], clean_plan)["all_exact"]
    changed = dict(pilot_clean, success=False)
    checks["changed_real_success_rejected"] = not clean_audit.paired_compare(clean_reference, changed,
        data["source_path"], data["pilot_path"])["all_exact"]
    checks["GPU_not_initialized"] = not torch.cuda.is_initialized()
    require(all(checks.values()), f"CPU checks failed: {checks}")
    report = {"status": "passed", "checks": checks, "count": len(checks),
              "script_sha256": sha(__file__), "config_sha256": sha(CONFIG),
              "frozen_old_selftest_reused": True, "GPU_used": False, "EGL_used": False,
              "learned_forward": False, "environment_constructed": False,
              "scope": "Synthetic routing/run_episode contracts plus read-only source/result/physics metadata verification; actual new Clean parity remains required at runtime."}
    require(not report_path.exists(), "Preserve existing CPU check evidence")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--cpu-out-dir", type=Path, default=CPU_OUT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    data = load_inputs(config)
    if args.self_test:
        print(json.dumps(cpu_self_test(config, data, args.cpu_out_dir / "self_test.json")), flush=True)
        return
    preflight = {"status": "passed", "script_sha256": sha(__file__), "config_sha256": sha(CONFIG),
                 "planned_episodes": data["planned"], "clean_source": str(data["source_path"]),
                 "clean_source_sha256": sha(data["source_path"]), "clean_successes": sum(r["success"] for r in data["clean_rows"].values()),
                 "known_pilots": 3, "new_extension": 27, "fresh_Clean_replication_required": True,
                 "GPU_used": False, "EGL_used": False, "learned_forward": False,
                 "all_registered_sources_and_artifacts_verified": True,
                 "packages": {k: importlib.metadata.version(k) for k in ("torch", "numpy", "Pillow", "imageio", "mujoco", "robosuite")}}
    if args.preflight:
        path = args.cpu_out_dir / "preflight.json"
        require(not path.exists(), "Preserve existing CPU preflight evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, preflight)
        print(json.dumps(preflight), flush=True)
        return
    cpu_path = args.cpu_out_dir / "self_test.json"
    checked = json.loads(cpu_path.read_text())
    require(checked["status"] == "passed" and checked["script_sha256"] == sha(__file__)
            and checked["config_sha256"] == sha(CONFIG), "Run matching CPU self-test before CUDA")
    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), "Output must be empty; no implicit resume or overwrite")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.cuda.set_per_process_memory_fraction(.28)
    provenance = copy.deepcopy(data["source"]["provenance"])
    provenance.update(runner_sha256=sha(__file__), frozen_intervention_runner_sha256=OLD_SHA,
        base_runner_sha256=sha(base.__file__), protocol_sha256=sha(CONFIG),
        clean_source_summary=str(data["source_path"]), clean_source_summary_sha256=sha(data["source_path"]),
        wrapper_source_sha256=config["source_sha256"], internal_future_query_context_intervention=True,
        external_attack_reachability_proven=False, goal_invariance_assumed=False, bus_cut_implemented=False,
        execution_offload=False, goal_routing="policy/value/environment original task Goal; only fixed-action future query receives cyclic next task Goal")
    started = time.monotonic()
    summary = {"status": "loading", "condition": "goal_swap", "experiment": "internal_future_query_goal_cyclic_matched30",
               "protocol": config["fixed_protocol"], "registration": config, "protocol_sha256": sha(CONFIG),
               "provenance": provenance, "planned_episodes": data["planned"], "results": [],
               "clean_reproduction": None, "decision": None, "decision_status": "incomplete_until_all30",
               "independent_analysis_required": True, "training_updates": 0, "api_requests": 0}
    path = out / "summary.json"
    write_json(path, summary)
    episode_args = SimpleNamespace(suite="libero_goal", best_of_n=2, seed=31, env_seed=0,
        max_steps=300, settle_steps=10, video_stride=2, offload=False, cuda_memory_fraction=.28)
    try:
        model, vae, sampler, rand = inference.load_model(device="cuda")
        require(old.expected_runtime_compatibility() == data["source"]["provenance"]["runtime_compatibility"],
                "Loaded runtime compatibility differs from Clean30")
        episode_args.out_dir = out / "clean_parity"
        episode_args.out_dir.mkdir()
        episode_args.swap_goal = data["goals"][1]
        task = data["suite"].get_task(0)
        fresh_clean = old.run_episode(model, vae, sampler, rand, task, 0, 0, data["texts"], data["stats"],
                                     episode_args, dict(provenance, execution_condition="clean"), "clean")
        clean_plan = dict(data["planned"][0], condition="clean", forecast_goal=data["goals"][0])
        comparison = compare_episode(data["clean_rows"][0, 0], fresh_clean, data["source_path"], path, clean_plan)
        summary["clean_reproduction"] = {"result": fresh_clean, "comparison": comparison}
        write_json(out / "clean_reproduction.json", summary["clean_reproduction"])
        write_json(path, summary)
        require(comparison["all_exact"], "Fresh wrapper Clean differs from full native Clean30 task0/init0")
        for item in data["planned"]:
            task_id, init_id = item["task_id"], item["init_id"]
            episode_args.out_dir = out
            episode_args.swap_goal = item["forecast_goal"]
            task = data["suite"].get_task(task_id)
            p = dict(provenance, execution_condition="goal_swap", original_task_id=task_id,
                     original_goal=item["goal"], swap_task_id=item["swap_task_id"], swap_goal=item["forecast_goal"])
            result = old.run_episode(model, vae, sampler, rand, task, task_id, init_id, data["texts"], data["stats"],
                                     episode_args, p, "goal_swap")
            validate_routing(result, item)
            for key in ("initial_state_sha256", "bddl_sha256", "init_file_sha256"):
                require(result[key] == data["clean_rows"][task_id, init_id][key], "Matched physical episode changed")
            result.update(swap_task_id=item["swap_task_id"], known_pilot_replication=item["known_pilot_replication"],
                          new_extension=item["new_extension"])
            if item["known_pilot_replication"]:
                result["known_pilot_reproduction"] = compare_episode(data["known"][task_id, init_id], result,
                    data["pilot_path"], path, item)
            # Retain every real result before checking reproduction/technical gates.
            summary["results"].append(result)
            write_json(out / f"libero_goal_task{task_id:02d}_init{init_id:02d}_goal_swap.json", result)
            summary.update(status="running", elapsed_seconds=time.monotonic()-started,
                           completed_episodes=len(summary["results"]))
            write_json(path, summary)
            if item["known_pilot_replication"]:
                require(result["known_pilot_reproduction"]["all_exact"], "Known pilot reproduction changed; technical review required")
            print(json.dumps({"completed_episodes": len(summary["results"]), "task_id": task_id,
                "init_id": init_id, "condition": "goal_swap", "success": result["success"],
                "environment_steps": result["environment_steps"], "decision": None}), flush=True)
            torch.cuda.empty_cache()
        require(set(keyed(summary["results"], "goal_swap")) == set(GRID), "Do not complete a partial grid")
        summary.update(status="completed", decision=None, decision_status="awaiting_independent_complete_grid_analysis")
    except BaseException as exc:
        summary.update(status="error", exception_type=type(exc).__name__, exception=str(exc),
                       decision=None, decision_status="incomplete_technical_abort_not_task_failure")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic()-started
        write_json(path, summary)
    print(json.dumps({"summary": str(path), "status": summary["status"], "completed_episodes": len(summary["results"]),
                      "decision": None}), flush=True)


if __name__ == "__main__":
    main()
