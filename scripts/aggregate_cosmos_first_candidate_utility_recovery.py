#!/usr/bin/env python3
"""CPU audit of the explicitly amended 50+10 Clean candidate-utility panel.

The interrupted original stays error. No panel utilities or uncertainty are
computed until all 60 branches and all 30 original-selected replays validate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import aggregate_cosmos_first_candidate_utility as old
import pilot_cosmos_first_candidate_utility_recovery as recovery

require, equal = old.require, old.equal


def snapshot(path):
    path = old.resolve(path)
    content = path.read_bytes()
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def validate_progress(doc, plan):
    """One attempt per remaining branch, in order; at most one pending/error."""
    require(doc.get("status") in ("running", "error", "completed"), "Unknown recovery status")
    rows, attempts = doc["results"], doc["attempts"]
    require(len(rows) <= 10 and len(rows) <= len(attempts) <= min(len(rows) + 1, 10), "Invalid attempt/result count")
    for index, attempt in enumerate(attempts):
        equal(attempt["index"], index, "Attempt order or duplicate")
        equal(attempt["branch"], plan[index], "Attempt changed the fixed missing branch")
        if index < len(rows):
            equal(attempt["status"], "validated", "Completed row lacks validated attempt")
            equal(attempt["exit_code"], 0, "Completed child exited abnormally")
            for key, value in plan[index].items():
                equal(rows[index].get(key), value, "Recovery rows not the registered prefix")
        else:
            require(attempt["status"] in ("invoked", "error", "returned"), "Invalid unfinished attempt")
    complete = doc["status"] == "completed"
    if complete:
        equal(len(rows), 10, "Completed recovery lacks ten validated branches")
        require("active_branch" not in doc, "Completed recovery still has active branch")
    elif attempts and "active_branch" in doc:
        equal(doc["active_branch"], plan[len(attempts) - 1], "Wrong active branch")
    return complete


def analyze(registration, summary_path):
    # Pure CPU preflight validates immutable inputs, source code, runtime protocol
    # and the six diagnostic artifacts. It does not create an environment.
    config, source, source_items, validation = recovery.preflight(registration)
    config_binding = old.file_record(registration)
    historical = old.analyze(config["original_registration"]["path"], config["original_run_summary"]["path"])
    equal(historical["status"], "incomplete", "Original v1 must remain incomplete")
    equal(historical["runner_status"], "error", "Original v1 terminal error was changed")
    equal(historical["validated_branches"], 50, "Original valid prefix changed")
    equal(historical["summary"], config["original_run_summary"], "Original summary binding changed")
    equal(historical["contribution"]["resource_gate"], None, "Original incomplete run cannot have a verdict")
    old_selected = sum(a["independent_source_comparison"]["selected_arm_whole_trajectory_exact"] is True
                       for a in historical["episode_audits"])
    equal(old_selected, 25, "Original selected-arm parity count")
    result = {
        "status": "incomplete", "execution_scope": recovery.EXECUTION_SCOPE,
        "analysis_code": old.file_record(__file__), "registration": config_binding,
        "original_registration": config["original_registration"],
        "original_run_summary": config["original_run_summary"],
        "original_incomplete_analysis": config["original_incomplete_analysis"],
        "rendering_diagnostic_summary": config["rendering_diagnostic_summary"],
        "source_summary": config["source_summary"], "original_runner_status": "error",
        "original_validated_branches": 50, "recovery_validated_branches": 0,
        "validated_branches": 50, "expected_branches": 60,
        "selected_arm_full_source_parity_count": 25, "complete_amended_grid": False,
        "episode_audits": [{**a, "execution_version": "original_interrupted_v1"}
                           for a in historical["episode_audits"]],
        "pairs": [], "contribution": old.summarize([], False),
        "gpu_used": False, "environment_created": False, "api_requests": 0, "training_updates": 0,
        "limits": [
            "Post-interruption amended development analysis; not completion of unchanged original v1 or a blind confirmation.",
            "The first missing branch already had an unmatched zero-action attempt; one new attempt per branch applies only within this recovery version.",
            "Six matching cold render diagnostics motivate engineering isolation but neither establish the old root cause nor prove its repair.",
            "Fresh processes and recorded parity do not establish equality of every unrecorded simulator/controller/renderer state for alternate branches.",
            "First-query selection under shared original-Goal N2 continuation; not forecast-only or every-query planning contribution.",
            "All30 original-selected outcomes and the old50 records were available before this amendment; uncertainty is descriptive for this fixed development panel.",
            "The original success-gain resource gate is unchanged; cost cannot rescue it. Earlier GoalSwap NoGo remains unchanged.",
        ],
    }
    if not old.resolve(summary_path).exists():
        result["reason"] = "Recovery summary absent; no missing utilities imputed"
        return result
    doc, summary_binding = snapshot(summary_path)
    result["recovery_summary"] = summary_binding
    result["recovery_runner_status"] = doc.get("status")
    equal(old.resolve(summary_path).parent, old.resolve(config["out_dir"]), "Wrong recovery output directory")
    equal(doc["config"], config_binding, "Recovery registration binding")
    equal(doc["runner_sha256"], config["runner"]["sha256"], "Recovery runner binding")
    for key in ("scope", "execution_scope", "execution_mode", "source_summary", "original_run_summary",
                "protocol", "resource_gate", "planned_recovery_branches", "execution", "api_requests", "training_updates"):
        equal(doc[key], config[key], "Recovery summary protocol: " + key)
    equal(doc["condition"], "first_candidate_utility_recovery", "Truthful recovery condition")
    expected_provenance = recovery.provenance(config, source, config_binding)
    equal(doc["provenance"], expected_provenance, "Recovery numerical provenance")
    completed = validate_progress(doc, config["planned_recovery_branches"])
    observed = {(a["task_id"], a["init_id"], a["forced_candidate_index"]): a
                for a in historical["episode_audits"]}
    parent_pids, child_pids = set(), set()
    for index, attempt in enumerate(doc["attempts"]):
        item = config["planned_recovery_branches"][index]
        directory = old.resolve(config["out_dir"]) / f"task{item['task_id']:02d}_init{item['init_id']:02d}" / f"candidate{item['forced_candidate_index']}"
        old.bound(attempt["reservation"]["path"], attempt["reservation"]["sha256"])
        equal(old.resolve(attempt["reservation"]["path"]), directory / "attempt_reservation.json", "Reservation escaped branch directory")
        reservation = old.read(attempt["reservation"]["path"])
        for key, value in {"index": index, "branch": item, "config": config_binding}.items():
            equal(reservation[key], value, "Attempt reservation: " + key)
        require(type(reservation["parent_pid"]) is int and reservation["parent_pid"] > 0, "Invalid parent PID")
        parent_pids.add(reservation["parent_pid"])
        command = attempt["command"]
        equal(command[1:], [config["runner"]["path"], "--child", "--child-index", str(index), "--config", config_binding["path"]], "Unregistered child command")
        for field in ("log", "child_summary"):
            if field in attempt:
                old.bound(attempt[field]["path"], attempt[field]["sha256"])
        if index >= len(doc["results"]):
            continue  # Retained failed/pending attempt is not a task failure or a utility.
        outer = doc["results"][index]
        child = old.read(attempt["child_summary"]["path"])
        equal(old.resolve(attempt["child_summary"]["path"]), directory / "child_summary.json", "Wrong child report directory")
        for key, value in {"status": "completed", "index": index, "branch": item, "config": config_binding}.items():
            equal(child[key], value, "Child completion proof: " + key)
        require(type(child["pid"]) is int and child["pid"] > 0 and child["pid"] not in child_pids, "Missing unique fresh child PID")
        child_pids.add(child["pid"])
        marker = old.read(directory / "child_started.json")
        for key, value in {"index": index, "pid": child["pid"], "parent_pid": reservation["parent_pid"]}.items():
            equal(marker[key], value, "Exclusive child marker: " + key)
        old.bound(child["result"]["path"], child["result"]["sha256"])
        equal(old.resolve(child["result"]["path"]), directory / "branch_result.json", "Wrong result directory")
        equal(old.read(child["result"]["path"]), outer, "Child result differs from summary")
        source_item = source_items[item["task_id"], item["init_id"]]
        recovery.audit_result(outer, item, source_item, expected_provenance, directory, old.resolve(summary_path))
        audited = old.audit_episode(outer["episode"], item["forced_candidate_index"])
        audited["independent_source_comparison"] = old.compare_with_source(
            outer["episode"], source_item["episode"], audited,
            old.audit_episode(source_item["episode"]), item["role"] == "native_selected")
        audited["execution_version"] = "fresh_process_recovery_v1"
        audited["attempt_artifacts"] = {"reservation": attempt["reservation"], "child_summary": attempt["child_summary"],
                                        "child_started": old.file_record(directory / "child_started.json"), "branch_result": child["result"]}
        key = item["task_id"], item["init_id"], item["forced_candidate_index"]
        require(key not in observed, "Recovery duplicates a valid original branch")
        observed[key] = audited
        result["episode_audits"].append(audited)
    require(len(parent_pids) <= 1, "Recovery attempted across multiple parent invocations")
    result["validated_branches"] = len(observed)
    result["recovery_validated_branches"] = len(doc["results"])
    result["attempt_count"] = len(doc["attempts"])
    result["selected_arm_full_source_parity_count"] = sum(
        a["independent_source_comparison"]["selected_arm_whole_trajectory_exact"] is True for a in observed.values())
    result["missing_branches"] = [{"task_id": t, "init_id": i, "forced_candidate_index": c}
                                  for t, i, c in old.GRID if (t, i, c) not in observed]
    if completed:
        equal(sorted(observed), old.GRID, "Amended full60 grid required")
        equal(result["selected_arm_full_source_parity_count"], 30, "All30 selected full-source parity required")
        result["pairs"] = [old.make_pair(t, i, source_items[t, i]["first_chunk"]["native_best_of_n"]["selected_index"],
                                          {c: observed[t, i, c] for c in (0, 1)}) for t, i in old.PANEL]
        result["complete_amended_grid"] = True
        result["status"] = "completed_amended_development_validated"
        result["contribution"] = old.summarize(result["pairs"], True)
        result["contribution"]["status"] = "complete_amended_development_panel"
    else:
        result["reason"] = "Not all new branches completed successfully; no panel utility, gate, CI or p-value computed"
    return result


def self_test():
    original, stopped, plan, _ = recovery.evidence()
    _, source_items, _ = recovery.frozen.load_clean_panel()
    require(stopped["status"] == "error" and len(stopped["results"]) == 50, "Historical error lost")
    partial = {"status": "running", "results": [], "attempts": []}
    checks = {"partial_no_completion": validate_progress(partial, plan) is False,
              "partial_no_statistics": old.summarize([], False)["resource_gate"] is None,
              "exact_missing_ten": plan == original["planned_branches"][50:],
              "actual_source_selection_schema": all(
                  source_items[item["task_id"], item["init_id"]]["first_chunk"]["native_best_of_n"]["selected_index"]
                  == item["source_native_selected_index"] for item in original["planned_branches"])}
    complete = {"status": "completed", "results": copy.deepcopy(plan), "attempts": [
        {"index": index, "branch": item, "status": "validated", "exit_code": 0}
        for index, item in enumerate(plan)]}
    checks["all_ten_complete_accepted"] = validate_progress(complete, plan) is True
    checks["all_ten_running_has_no_verdict"] = validate_progress({**complete, "status": "running"}, plan) is False
    def rejected(doc):
        try: validate_progress(doc, plan)
        except (ValueError, KeyError): return True
        return False
    checks["completed_missing_branches_rejected"] = rejected({**partial, "status": "completed"})
    duplicate = {"status": "running", "results": [dict(plan[0])], "attempts": [
        {"index": 0, "branch": plan[0], "status": "validated", "exit_code": 0},
        {"index": 0, "branch": plan[0], "status": "invoked"}]}
    checks["duplicate_attempt_rejected"] = rejected(duplicate)
    reordered = copy.deepcopy(duplicate)
    reordered["attempts"][1].update(index=1, branch=plan[2])
    checks["skipped_branch_rejected"] = rejected(reordered)
    checks["no_model_or_environment_import"] = "torch" not in sys.modules and "libero.libero.envs" not in sys.modules
    require(all(checks.values()), "CPU audit checks failed: " + repr(checks))
    return {"status": "passed", "checks": checks, "gpu_used": False, "environment_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if not all((args.registration, args.summary, args.out_dir)):
        parser.error("--registration, --summary and --out-dir required")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze(args.registration, args.summary)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "invalid", "execution_scope": recovery.EXECUTION_SCOPE,
                  "error": {"type": type(error).__name__, "message": str(error)},
                  "analysis_code": old.file_record(__file__), "pairs": [], "contribution": None,
                  "gpu_used": False, "environment_created": False}
    with (out / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write("\n")
    contribution = result.get("contribution") or {}
    tables = {"paired_results.csv": result["pairs"], "task_results.csv": contribution.get("per_task", []),
              "policy_results.csv": [{"policy": k, **v} for k, v in contribution.get("policies", {}).items()]}
    for filename, rows in tables.items():
        with (out / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": result["status"], "analysis": str(out / "analysis.json"),
                      "resource_gate": contribution.get("resource_gate"), "error": result.get("error")}))
    sys.exit(0 if result["status"] == "completed_amended_development_validated" else 2)


if __name__ == "__main__":
    main()
