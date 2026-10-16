#!/usr/bin/env python3
"""Prospectively bounded, fresh-process completion of ten missing Clean branches.

The interrupted v1 stays terminal and immutable. This program never retries a
child, changes a candidate, relaxes a parity check, or computes a scientific gate.
Preparation, preflight, parent orchestration and self-tests do not import models.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import pilot_cosmos_first_candidate_utility as frozen
import aggregate_cosmos_first_candidate_utility as old_analysis

ROOT = frozen.ROOT
CONFIG = ROOT / "configs/cosmos_first_candidate_utility_recovery_v1.json"
OUT = ROOT / "runs/cosmos_first_candidate_utility_recovery_v1"
OLD_CONFIG = frozen.DEFAULT_CONFIG
OLD_SUMMARY = frozen.DEFAULT_OUT / "summary.json"
OLD_ANALYSIS = ROOT / "runs/cosmos_first_candidate_utility_matched30_incomplete_audit_v1/analysis.json"
DIAGNOSTIC = ROOT / "runs/cosmos_rendering_reproducibility_v1/summary.json"
FAILED_DIR = frozen.DEFAULT_OUT / "task08_init01/candidate1"
AGGREGATOR = ROOT / "scripts/aggregate_cosmos_first_candidate_utility_recovery.py"
KNOWN = {
    OLD_CONFIG: "2fd04118a35fa4785eb725264970a41525248c5b0a332955c99b27cba97dbd23",
    OLD_SUMMARY: "7648a0edb1d8c151b3a4cce1a3223a22879bfe187041319d9a4bad1c770e5a47",
    OLD_ANALYSIS: "810590bbaeb614917893ced2bf9ab4ea3153a92aeeb11a5e6ef763e05c09527d",
    DIAGNOSTIC: "7078b7f1cf8e525a267d3620f8ffe413362575b6bbc4992dd57f155b078dc56d",
}
SCOPE = frozen.SCOPE
EXECUTION_SCOPE = "post_interruption_amended_development_analysis"
EXECUTION_MODE = "fresh_process_recovery_single_attempt"
EXECUTION = {
    "maximum_new_child_attempts": 10, "attempts_per_registered_branch": 1,
    "fresh_process_per_branch": True, "child_timeout_seconds": 900,
    "parallel_children": 1, "on_any_error": "terminal; no replacement, retry or remaining branches",
    "prior_unmatched_attempt_retained": True,
    "scientific_protocol": "Original candidate order, original Goal, seed31/32, N2 continuation, 300-step budget and exact checks unchanged",
}
require, sha, read, write, binding = frozen.require, frozen.sha, frozen.read, frozen.write, frozen.binding


def bound(item):
    require(binding(item["path"]) == {k: item[k] for k in ("path", "sha256")}, "Artifact binding changed: " + item["path"])


def remaining_plan(plan, completed):
    require(len(plan) == 60 and len(completed) == 50, "Recovery is only the registered 50+10 split")
    for expected, actual in zip(plan, completed):
        require(all(actual.get(k) == v for k, v in expected.items()), "Original completed rows are not the exact plan prefix")
    remaining = copy.deepcopy(plan[50:])
    require(len(remaining) == 10 and len({(x["task_id"], x["init_id"], x["forced_candidate_index"]) for x in remaining}) == 10, "Invalid missing branch grid")
    for left, right in zip(remaining[::2], remaining[1::2]):
        require(left["role"] == "native_selected" and right["role"] == "alternate"
                and (left["task_id"], left["init_id"]) == (right["task_id"], right["init_id"])
                and left["forced_candidate_index"] == left["source_native_selected_index"]
                and right["forced_candidate_index"] == 1 - left["forced_candidate_index"], "Selected-first recovery pair changed")
    return remaining


def evidence():
    for path, expected in KNOWN.items():
        require(sha(path) == expected, "Fixed historical evidence changed: " + str(path))
    original, _ = read(OLD_CONFIG)
    stopped, _ = read(OLD_SUMMARY)
    audited, _ = read(OLD_ANALYSIS)
    diagnostic, _ = read(DIAGNOSTIC)
    require(stopped["status"] == "error" and stopped["exception"] == "Initial observation parity failed", "Original interruption changed")
    require(audited["status"] == "incomplete" and audited["validated_branches"] == 50
            and audited["contribution"]["resource_gate"] is None
            and audited["contribution"]["task_cluster_uncertainty"] is None, "Original incomplete audit changed")
    plan = remaining_plan(original["planned_branches"], stopped["results"])
    require(stopped["active_branch"] == plan[0], "Original failed attempt is not the first missing branch")
    failure, _ = read(FAILED_DIR / "initial_parity.json")
    require(failure["all_exact"] is False and failure["checks"] == {
        "initial_dtype_shape_bytes": True, "settled_dtype_shape_bytes": True,
        "initial_png_bytes": False, "first_eef_position": True}, "Original mismatch evidence changed")
    failed_ep, _ = read(FAILED_DIR / "libero_goal_task08_init01_clean.json")
    require(failed_ep["status"] == "error" and failed_ep["environment_steps"] == failed_ep["action_chunks"] == 0,
            "Unmatched attempt unexpectedly contains actions")
    require(diagnostic["status"] == "completed_diagnostic" and len(diagnostic["trials"]) == 6, "Bounded diagnostic is incomplete")
    for trial in diagnostic["trials"]:
        bound(trial["result"])
        result, _ = read(trial["result"]["path"])
        require(trial["exit_code"] == 0 and result["status"] == "completed_diagnostic"
                and result["model_forward_calls"] == result["policy_steps"] == 0,
                "Diagnostic execution changed")
        require(result["source_image_comparison"]["png_bytes_exact"] is True
                and all(result["source_state_comparison"][k] is True for k in
                        ("initial_sim_exact", "settled_sim_exact", "settled_eef_exact")), "Cold diagnostic source parity changed")
    paths = set(KNOWN)
    paths.update(p for p in DIAGNOSTIC.parent.rglob("*") if p.is_file())
    paths.update(p for p in FAILED_DIR.rglob("*") if p.is_file())
    paths.add(ROOT / "runs/cosmos_initial_observation_parity_cpu_v1/analysis.json")
    for outer in stopped["results"]:
        for key in ("episode_json", "source_episode_json", "first_query_parity"):
            bound(outer[key]); paths.add(Path(outer[key]["path"]))
        for item in outer["artifacts"].values():
            bound(item); paths.add(Path(item["path"]))
        directory = Path(outer["episode_json"]["path"]).parent
        paths.add(directory / "initial_parity.json")
        if outer["role"] == "native_selected":
            require(outer["selected_arm_parity"]["all_exact"] is True, "Previously completed selected arm failed")
            paths.add(directory / "selected_arm_parity.json")
    return original, stopped, plan, {str(p.resolve()): sha(p) for p in sorted(paths)}


def source_inventory(original):
    sources = dict(original["source_sha256"])
    for path in (Path(__file__), AGGREGATOR):
        sources[frozen.rel(path)] = sha(path)
    return sources


def prepare(path, out):
    original, source, rows, _ = frozen.preflight(OLD_CONFIG)
    _, stopped, plan, references = evidence()
    config = {
        "schema_version": 1, "status": "prospective_clean_only_recovery_registration",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE, "execution_scope": EXECUTION_SCOPE, "execution_mode": EXECUTION_MODE,
        "runner": binding(__file__), "out_dir": str(Path(out).resolve()),
        "original_registration": binding(OLD_CONFIG), "original_run_summary": binding(OLD_SUMMARY),
        "original_incomplete_analysis": binding(OLD_ANALYSIS), "rendering_diagnostic_summary": binding(DIAGNOSTIC),
        "source_summary": original["source_summary"], "protocol": copy.deepcopy(frozen.PROTOCOL),
        "resource_gate": copy.deepcopy(frozen.RESOURCE_GATE), "execution": EXECUTION,
        "planned_recovery_branches": plan, "original_valid_branch_count": 50,
        "source_sha256": source_inventory(original), "reference_sha256": original["reference_sha256"],
        "physical_file_sha256": original["physical_file_sha256"], "model_sha256": original["model_sha256"],
        "historical_evidence_sha256": references,
        "original_failed_attempt": {"branch": stopped["active_branch"], "directory": str(FAILED_DIR),
            "episode": binding(FAILED_DIR / "libero_goal_task08_init01_clean.json"),
            "initial_parity": binding(FAILED_DIR / "initial_parity.json"),
            "initial_image": binding(FAILED_DIR / "libero_goal_task08_init01_clean_initial.png"),
            "interpretation": "Recorded initial/settled sim arrays and EEF matched; wrist image had three differing RGB pixels before the first model query. The unmatched attempt remains in the evidence; no control or task outcome was measured there."},
        "amendment": [
            "Original v1 remains error with 50 valid results; its failure and incomplete audit are never rewritten.",
            "The only execution change is one fresh process/model/environment per remaining branch instead of a shared long-lived model process. All ten are fixed before execution, including the prior unmatched branch, with one new attempt each.",
            "All six no-model cold-process render diagnostics matched the source. This does not identify the prior mismatch's root cause or guarantee a fresh model process will reproduce it. Every original exact check remains mandatory.",
            "The prior selected branch must reproduce its entire source trajectory before its alternate is permitted. Any failure terminates this recovery; no replacement or further branch is run.",
            "Joining the old50 and new10 is explicitly a post-interruption amended development analysis. These are not a fresh blind panel or completion of the unchanged original v1.",
            "Only normal original-Goal ranking contribution is measured. Earlier GoalSwap NoGo and other failed gates remain in force, independent of this result.",
            "No controller/RNG completeness is inferred from the stored sim array. The actual contract is identical reset/settle, recorded initial arrays/images, regenerated candidates and full selected-arm trajectory parity.",
        ],
        "runtime_requirements": original["runtime_requirements"], "api_requests": 0, "training_updates": 0,
    }
    require(Path(out).resolve() != frozen.DEFAULT_OUT.resolve(), "Cannot reuse original output")
    write(path, config, exclusive=True)
    return binding(path)


def preflight(path):
    config, digest = read(path)
    require(config["schema_version"] == 1 and config["status"] == "prospective_clean_only_recovery_registration", "Wrong recovery registration")
    for key, value in {"scope": SCOPE, "execution_scope": EXECUTION_SCOPE, "execution_mode": EXECUTION_MODE,
                       "protocol": frozen.PROTOCOL, "resource_gate": frozen.RESOURCE_GATE,
                       "execution": EXECUTION, "original_valid_branch_count": 50,
                       "api_requests": 0, "training_updates": 0}.items():
        require(config[key] == value, "Recovery protocol differs: " + key)
    require(config["runner"] == binding(__file__), "Recovery runner changed after registration")
    for key, filename in (("original_registration", OLD_CONFIG), ("original_run_summary", OLD_SUMMARY),
                          ("original_incomplete_analysis", OLD_ANALYSIS), ("rendering_diagnostic_summary", DIAGNOSTIC)):
        require(config[key] == binding(filename), "Historical input binding changed: " + key)
    original, source, rows, _ = frozen.preflight(OLD_CONFIG)
    _, _, plan, references = evidence()
    require(config["planned_recovery_branches"] == plan and config["historical_evidence_sha256"] == references, "Recovery plan/evidence changed")
    require(config["source_sha256"] == source_inventory(original), "Recovery source inventory changed")
    for key in ("source_summary", "reference_sha256", "physical_file_sha256", "model_sha256", "runtime_requirements"):
        require(config[key] == original[key], "Original input differs: " + key)
    require(Path(config["out_dir"]).resolve() not in (frozen.DEFAULT_OUT, DIAGNOSTIC.parent), "Cannot reuse historical output")
    validation = {"status": "cpu_preflight_passed", "config": {"path": str(Path(path).resolve()), "sha256": digest},
                  "runner_sha256": sha(__file__), "original_valid_branches": 50, "new_branches": 10,
                  "source_files": len(config["source_sha256"]), "historical_artifacts": len(references),
                  "GPU_used": False, "environment_created": False, "api_requests": 0, "training_updates": 0}
    return config, source, rows, validation


def provenance(config, source, config_binding):
    result = copy.deepcopy(source["provenance"])
    result["first_candidate_utility_wrapper"] = {
        "config": config_binding, "runner_sha256": config["runner"]["sha256"], "scope": SCOPE,
        "execution_mode": EXECUTION_MODE, "original_interrupted_run": config["original_run_summary"],
        "first_selection": "registered original candidate index; native original Goal N2 continuation thereafter",
    }
    return result


def next_branch(plan, results):
    require(len(results) <= len(plan), "Too many completed recovery branches")
    for expected, actual in zip(plan, results):
        require(all(actual.get(k) == v for k, v in expected.items()), "Recovery result is not the plan prefix")
        if expected["role"] == "native_selected":
            require(actual.get("selected_arm_parity", {}).get("all_exact") is True, "Selected-arm parity gate failed")
    return None if len(results) == len(plan) else plan[len(results)]


def audit_result(result, item, source_item, expected_provenance, directory, summary_path):
    require(all(result.get(k) == v for k, v in item.items()) and result["condition"] == "first_candidate_utility", "Wrong recovered branch")
    row = result["episode"]
    require(row["provenance"] == expected_provenance, "Recovery provenance differs")
    for key in ("trace", "chunks", "initial_image", "final_image"):
        require(Path(row[key]).resolve().parent == directory.resolve(), "Episode artifact escaped its new branch directory")
    bound(result["episode_json"])
    require(read(result["episode_json"]["path"])[0] == row, "Episode JSON differs from branch result")
    require(result["source_episode_json"] == binding(source_item["episode_json"]), "Wrong source episode")
    require(result["artifacts"] == frozen.audit_branch(row, source_item, item["forced_candidate_index"], directory), "Branch artifact inventory differs")
    initial, _ = read(directory / "initial_parity.json")
    require(result["initial_parity"] == initial and initial["all_exact"] is True
            and all(v is True for v in initial["checks"].values()), "Initial parity failed")
    bound(result["first_query_parity"])
    proof, _ = read(result["first_query_parity"]["path"])
    require(proof["all_exact"] is True and all(v is True for v in proof["checks"].values())
            and proof["forced_candidate_index"] == item["forced_candidate_index"]
            and proof["source_native_selected_index"] == item["source_native_selected_index"]
            and proof["source_chunks"] == binding(source_item["episode"]["chunks"]), "First query parity failed")
    current_audit = old_analysis.audit_episode(row, item["forced_candidate_index"])
    source_audit = old_analysis.audit_episode(source_item["episode"])
    old_analysis.compare_with_source(row, source_item["episode"], current_audit, source_audit, item["role"] == "native_selected")
    if item["role"] == "native_selected":
        checked = frozen.audit.paired_compare(source_item["episode"], row, frozen.SOURCE, summary_path)
        checked["trace_file_bytes_exact"] = sha(source_item["episode"]["trace"]) == sha(row["trace"])
        checked["all_exact"] &= checked["trace_file_bytes_exact"]
        require(checked["all_exact"] is True and result["selected_arm_parity"] == checked
                and read(directory / "selected_arm_parity.json")[0] == checked, "Selected full trajectory parity failed")
    else:
        require(result["selected_arm_parity"] is None, "Alternate cannot claim full selected parity")


def child(config_path, index):
    # The parent reserves this directory once. The exclusive marker forbids a
    # second invocation even after a crash or failed check in this child.
    initial_config, _ = read(config_path)
    require(type(index) is int and 0 <= index < 10, "Invalid child index")
    item = initial_config["planned_recovery_branches"][index]
    directory = Path(initial_config["out_dir"]) / f"task{item['task_id']:02d}_init{item['init_id']:02d}" / f"candidate{item['forced_candidate_index']}"
    reservation, _ = read(directory / "attempt_reservation.json")
    require(reservation["branch"] == item and reservation["index"] == index
            and reservation["parent_pid"] == os.getppid(), "Child was not launched by its registered parent")
    write(directory / "child_started.json", {"index": index, "pid": os.getpid(), "parent_pid": os.getppid(),
          "started_at_utc": datetime.now(timezone.utc).isoformat()}, exclusive=True)
    report = {"status": "starting", "index": index, "branch": item, "pid": os.getpid(), "result": None}
    started = time.monotonic()
    try:
        config, source, rows, validation = preflight(config_path)
        require(reservation["config"] == validation["config"], "Child registration differs from reservation")
        report["config"] = validation["config"]
        import torch
        import pilot_cosmos_policy_native_task_gate as native
        from libero.libero import benchmark, envs
        inference = native.inference
        require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1" and os.environ.get("MUJOCO_GL") == "egl", "Runtime backend differs")
        require(torch.__version__ == source["provenance"]["torch_version"] and np.__version__ == source["provenance"]["numpy_version"], "Torch/numpy versions differ")
        require(torch.cuda.is_available(), "CUDA required")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(frozen.PROTOCOL["cuda_memory_fraction"])
        with (inference.MODEL.parent / "libero_t5_embeddings.pkl").open("rb") as handle:
            text = pickle.load(handle)
        stats = inference.load_stats(inference.MODEL.parent / "libero_dataset_statistics.json")
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        model, vae, sampler, rand = inference.load_model(device="cuda")
        from cosmos_policy_torch_compat import compatibility_metadata
        require(compatibility_metadata() == source["provenance"]["runtime_compatibility"], "Runtime compatibility differs")
        task_id, init_id, forced = item["task_id"], item["init_id"], item["forced_candidate_index"]
        source_item = rows[task_id, init_id]
        task = suite.get_task(task_id)
        require(task.language == source_item["episode"]["goal"] and task.name == source_item["episode"]["task_name"], "Original task mapping changed")
        args = SimpleNamespace(**{k: frozen.PROTOCOL[k] for k in ("suite", "seed", "env_seed", "settle_steps", "max_steps", "best_of_n", "video_stride")}, out_dir=directory)
        numerical_provenance = provenance(config, source, validation["config"])
        report["status"] = "running"
        write(directory / "child_summary.json", report)
        with frozen.first_query_override(inference, envs, source_item, forced, directory) as tracker:
            episode = native.run_episode(model, vae, sampler, rand, task, task_id, init_id, text, stats, args, numerical_provenance)
        require(tracker["first_query"] is not None and tracker["first_query"]["all_exact"], "Missing regenerated first query")
        artifacts = frozen.audit_branch(episode, source_item, forced, directory)
        selected = None
        if item["role"] == "native_selected":
            selected = frozen.audit.paired_compare(source_item["episode"], episode, frozen.SOURCE, Path(config["out_dir"]) / "summary.json")
            selected["trace_file_bytes_exact"] = sha(source_item["episode"]["trace"]) == sha(episode["trace"])
            selected["all_exact"] &= selected["trace_file_bytes_exact"]
            write(directory / "selected_arm_parity.json", selected)
            require(selected["all_exact"], "Original-selected full trajectory parity failed")
        result = {**item, "condition": "first_candidate_utility", "episode": episode,
                  "episode_json": binding(Path(episode["trace"]).with_suffix(".json")),
                  "source_episode_json": binding(source_item["episode_json"]),
                  "first_query_parity": {**binding(directory / "first_query_parity.json"), "all_exact": True},
                  "initial_parity": read(directory / "initial_parity.json")[0],
                  "selected_arm_parity": selected, "artifacts": artifacts}
        audit_result(result, item, source_item, numerical_provenance, directory, Path(config["out_dir"]) / "summary.json")
        write(directory / "branch_result.json", result, exclusive=True)
        report.update(status="completed", result=binding(directory / "branch_result.json"))
    except BaseException as error:
        report.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write(directory / "child_summary.json", report)


def invoke(command, log, timeout):
    """Exactly one process. Timeout/interrupt kills its process group, no retry."""
    with Path(log).open("x") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def run(config_path):
    config, source, rows, validation = preflight(config_path)
    output = Path(config["out_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "cpu_preflight.json", validation, exclusive=True)
    numerical_provenance = provenance(config, source, validation["config"])
    summary = {"schema_version": 1, "status": "running", "condition": "first_candidate_utility_recovery",
               "scope": SCOPE, "execution_scope": EXECUTION_SCOPE, "execution_mode": EXECUTION_MODE,
               "config": validation["config"], "runner_sha256": sha(__file__),
               "source_summary": config["source_summary"], "original_run_summary": config["original_run_summary"],
               "protocol": config["protocol"], "resource_gate": config["resource_gate"],
               "planned_recovery_branches": config["planned_recovery_branches"], "execution": EXECUTION,
               "provenance": numerical_provenance, "results": [], "attempts": [],
               "api_requests": 0, "training_updates": 0}
    summary_path = output / "summary.json"
    started = time.monotonic()
    write(summary_path, summary)
    try:
        while (item := next_branch(config["planned_recovery_branches"], summary["results"])) is not None:
            index = len(summary["results"])
            directory = output / f"task{item['task_id']:02d}_init{item['init_id']:02d}" / f"candidate{item['forced_candidate_index']}"
            directory.mkdir(parents=True, exist_ok=False)
            reservation = {"index": index, "branch": item, "parent_pid": os.getpid(), "config": validation["config"]}
            write(directory / "attempt_reservation.json", reservation, exclusive=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--child", "--child-index", str(index), "--config", str(Path(config_path).resolve())]
            log = directory / "child.log"
            attempt = {"index": index, "branch": item, "status": "invoked", "command": command,
                       "reservation": binding(directory / "attempt_reservation.json")}
            summary["active_branch"] = item
            summary["attempts"].append(attempt)
            write(summary_path, summary)
            try:
                code = invoke(command, log, EXECUTION["child_timeout_seconds"])
                attempt.update(status="returned", exit_code=code)
                if (directory / "child_summary.json").exists():
                    attempt["child_summary"] = binding(directory / "child_summary.json")
                require(code == 0, f"Child {index} failed with exit {code}; recovery terminal")
                child_report, _ = read(directory / "child_summary.json")
                require(child_report["status"] == "completed" and child_report["index"] == index
                        and child_report["branch"] == item and child_report["config"] == validation["config"], "Invalid child completion")
                bound(child_report["result"])
                result, _ = read(child_report["result"]["path"])
                audit_result(result, item, rows[item["task_id"], item["init_id"]], numerical_provenance, directory, summary_path)
                summary["results"].append(result)
                attempt["status"] = "validated"
            except BaseException as error:
                attempt.update(status="error", exception_type=type(error).__name__, exception=str(error))
                raise
            finally:
                if log.exists():
                    attempt["log"] = binding(log)
                summary["elapsed_seconds"] = time.monotonic() - started
                write(summary_path, summary)
            print(json.dumps({"completed_recovery_branches": len(summary["results"]), **item,
                              "success": result["episode"]["success"], "steps": result["episode"]["environment_steps"]}), flush=True)
        require(len(summary["results"]) == len(summary["attempts"]) == 10, "Incomplete recovery")
        # Recheck immutable old evidence before declaring only this new run done.
        evidence()
        summary["status"] = "completed"
        summary.pop("active_branch", None)
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write(summary_path, summary)


def self_test():
    original, stopped, plan, _ = evidence()
    checks = {"only_missing10_original_order": plan == original["planned_branches"][50:],
              "old_status_remains_error": stopped["status"] == "error",
              "physical_protocol_reused": frozen.PROTOCOL["max_steps"] == 300 and frozen.PROTOCOL["best_of_n"] == 2
                  and frozen.PROTOCOL["seed"] == 31 and frozen.PROTOCOL["candidate_seeds"] == [31, 32]
                  and frozen.PROTOCOL["cuda_memory_fraction"] == .28 and frozen.PROTOCOL["offload"] is False,
              "original_gate_reused": frozen.RESOURCE_GATE["minimum_native_minus_uniform_success_rate"] == .05
                  and frozen.RESOURCE_GATE["minimum_tasks_with_positive_native_minus_alternate_net_successes"] == 2,
              "no_gpu_or_environment_import": "torch" not in sys.modules and "libero.libero.envs" not in sys.modules}
    def rejected(fn):
        try: fn()
        except (ValueError, KeyError): return True
        return False
    checks["missing_old_branch_rejected"] = rejected(lambda: remaining_plan(original["planned_branches"], stopped["results"][:-1]))
    changed = copy.deepcopy(stopped["results"])
    changed[-1]["forced_candidate_index"] ^= 1
    checks["altered_old_prefix_rejected"] = rejected(lambda: remaining_plan(original["planned_branches"], changed))
    checks["bad_selected_blocks_alternate"] = rejected(lambda: next_branch(plan, [{**plan[0], "selected_arm_parity": {"all_exact": False}}]))
    checks["missing_selected_proof_blocks_alternate"] = rejected(lambda: next_branch(plan, [dict(plan[0])]))
    checks["passed_selected_allows_original_alternate"] = next_branch(plan, [{**plan[0], "selected_arm_parity": {"all_exact": True}}]) == plan[1]
    checks["no_skip_allowed"] = rejected(lambda: next_branch(plan, [dict(plan[1])]))
    completed = [{**x, "selected_arm_parity": {"all_exact": True} if x["role"] == "native_selected" else None} for x in plan]
    checks["exact10_stops"] = next_branch(plan, completed) is None
    with tempfile.TemporaryDirectory(prefix="cosmos_recovery_cpu_") as temp:
        temp = Path(temp)
        # This runs only tiny CPU Python processes, not a model or environment.
        count = temp / "count"
        code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text()+'x' if p.exists() else 'x'); sys.exit(7)"
        returned = invoke([sys.executable, "-c", code, str(count)], temp / "mock.log", 5)
        checks["failed_child_invoked_exactly_once"] = returned == 7 and count.read_text() == "x"
        marker = temp / "marker.json"
        write(marker, {"attempt": 1}, exclusive=True)
        try: write(marker, {"attempt": 2}, exclusive=True)
        except FileExistsError: checks["exclusive_marker_rejects_restart"] = read(marker)[0] == {"attempt": 1}
        else: checks["exclusive_marker_rejects_restart"] = False
    require(all(checks.values()), "CPU self-test failed: " + repr(checks))
    return {"status": "passed", "checks": checks, "GPU_used": False, "environment_created": False,
            "model_imported": False, "api_requests": 0, "training_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-registration", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.prepare_registration: result = prepare(args.config, args.out_dir)
    elif args.preflight: result = preflight(args.config)[3]
    elif args.self_test: result = self_test()
    elif args.child: child(args.config, args.child_index); return
    else: run(args.config); return
    if args.report: write(args.report, result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
