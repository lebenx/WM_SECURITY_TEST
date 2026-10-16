#!/usr/bin/env python3
"""Recompute the matched ordinary-controller comparison from archived traces only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_dino_metaworld_clean_gate import bound, check, close, setup_arrays, sha, trace_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/dino_metaworld_servo_baseline_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dino_metaworld_servo_analysis_v1"))
    args = parser.parse_args()
    source = args.run_dir / "summary.json"
    summary = json.loads(source.read_text())
    check(summary["status"] == "completed", "All fixed episodes must finish")
    config = json.loads(bound(summary["config"]).read_text())
    clean = json.loads(bound(config["source_summary"]).read_text())
    clean_analysis = json.loads(bound(config["source_analysis"]).read_text())
    bound(config["source_config"])
    check(clean["config"] == config["source_config"], "Clean configuration linkage differs")
    shared_auditor = Path(__file__).with_name("analyze_dino_metaworld_clean_gate.py").resolve()
    check(sha(shared_auditor) == config["source_sha256"][str(shared_auditor)], "Shared trace auditor changed")
    check(clean_analysis["source"] == config["source_summary"] == summary["source_summary"], "Wrong Clean reference")
    check(summary["runner"] == config["runner"] and summary["frozen_helper"] == config["frozen_helper"], "Changed runner identity")
    check(summary["protocol"] == config["protocol"], "Summary protocol differs")
    bound(summary["runner"]); bound(summary["frozen_helper"])
    check(clean_analysis["status"] == "completed_independent_trace_audit", "Missing Clean audit")
    for record in (summary, clean):
        check(record["status"] == "completed" and [e["seed"] for e in record["episodes"]] == [1, 2, 3], "Incomplete or changed panel")
    check(not summary["cuda_initialized"] and summary["model_calls"] == summary["model_loads"] == 0, "Unexpected model/GPU use")
    check(summary["api_requests"] == summary["training_updates"] == 0, "Unexpected API/training work")
    expected_elapsed = [1, 16, 31, 46, 61, 76, 91]
    holds = [15] * 6 + [9]
    check(config["protocol"]["query_elapsed_steps"] == expected_elapsed
          and config["protocol"]["hold_control_counts"] == holds
          and config["protocol"]["gain"] == 5.0 and config["protocol"]["gripper"] == 0.0,
          "Registered controller or schedule changed")
    results = []
    for servo, original in zip(summary["episodes"], clean["episodes"]):
        check(servo["status"] == original["status"] == "completed", "Episode incomplete")
        check(servo["seed"] == original["seed"], "Seed mismatch")
        a = setup_arrays(servo["setup"])
        b = setup_arrays(original["setup"])
        check(a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a), "Different initial state or goal")
        for key in ("initial_image", "goal_image"):
            check(servo["setup"][key]["sha256"] == original["setup"][key]["sha256"], "Different input image")
        servo_result, rows = trace_outcome(servo["outcome"], servo["setup"], a, False)
        clean_result, _ = trace_outcome(original["outcome"], original["setup"], b, False)
        decisions = servo["decisions"]
        check(json.loads(bound(servo["queries_file"]).read_text()) == decisions, "Decision artifact differs")
        check([q["elapsed_before"] for q in decisions] == expected_elapsed, "Different feedback schedule")
        check([q["executed_control_count"] for q in decisions] == holds, "Different control budget")
        goal = a["goal_proprio"][-1]
        initial_proprio = a["initial_proprio"][-1]
        for vector in (goal, initial_proprio):
            check(vector.shape == (4,) and vector.dtype == np.float32 and np.isfinite(vector).all(),
                  "Invalid raw proprio vector")
        check(np.array_equal(initial_proprio, a["initial_state"][:4]), "Initial proprio/state differ")
        offset = 0
        for index, (decision, hold) in enumerate(zip(decisions, holds), 1):
            # Independently recover each policy input from the last real observation.
            current = (a["initial_proprio"][-1] if offset == 0 else np.asarray(rows[offset - 1]["state"][:4], dtype=np.float32))
            check(current.shape == goal.shape == (4,), "Unexpected proprio dimension")
            check(current.dtype == np.float32 and np.isfinite(current).all(), "Invalid current proprio")
            if offset:
                check(current.tolist() == rows[offset - 1]["state"][:4], "Current state is not exact float32 proprio")
            expected = np.zeros(4, dtype=np.float32)
            expected[:3] = np.minimum(np.maximum((goal[:3] - current[:3]) * np.float32(5), -1), 1)
            check(decision["current_proprio"] == current.tolist() and decision["goal_proprio"] == goal.tolist(), "Policy input differs")
            check(decision["action"] == expected.tolist(), "Controller rule differs")
            for step, row in enumerate(rows[offset:offset + hold]):
                check(row["action"] == expected.tolist() and row["planning_call"] == index
                      and row["action_in_chunk"] == step, "Action was not held for the registered interval")
            offset += hold
        check(offset == 99, "Incomplete physical trace")
        results.append({"seed": servo["seed"],
                        "clean_final_success": clean_result["final_success"],
                        "servo_final_success": servo_result["final_success"],
                        "clean_final_distance": clean_result["final_reach_distance"],
                        "servo_final_distance": servo_result["final_reach_distance"],
                        "servo_minus_clean_distance": servo_result["final_reach_distance"] - clean_result["final_reach_distance"],
                        "clean_return": clean_result["reward_sum"], "servo_return": servo_result["reward_sum"],
                        "servo_minus_clean_return": servo_result["reward_sum"] - clean_result["reward_sum"]})
    n_clean = sum(r["clean_final_success"] for r in results)
    n_servo = sum(r["servo_final_success"] for r in results)
    covers = all(not r["clean_final_success"] or r["servo_final_success"] for r in results)
    comparison = summary["comparison"]
    check(comparison["denominator"] == 3 and len(comparison["paired_results"]) == 3,
          "Saved comparison panel incomplete")
    check([r["seed"] for r in comparison["paired_results"]] == [1, 2, 3], "Saved comparison seed order differs")
    check(n_clean == comparison["clean_successes"] and n_servo == comparison["servo_successes"], "Success counts differ")
    check(covers == comparison["servo_covers_clean_successes"], "Coverage differs")
    no_extra_benefit = covers and n_servo >= n_clean
    check(no_extra_benefit == comparison["no_observed_additional_wm_task_success_benefit_over_servo"], "Interpretation rule differs")
    for observed, saved in zip(results, comparison["paired_results"]):
        for key, value in saved.items():
            if isinstance(value, float): close(observed[key], value)
            else: check(observed[key] == value, "Saved comparison differs")
    report = {
        "status": "completed_independent_servo_comparison_audit",
        "source": {"path": str(source.resolve()), "sha256": sha(source)},
        "clean_analysis": config["source_analysis"],
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "shared_trace_audit_code": {"path": str(Path(__file__).with_name("analyze_dino_metaworld_clean_gate.py").resolve()),
                                    "sha256": sha(Path(__file__).with_name("analyze_dino_metaworld_clean_gate.py"))},
        "episodes": results, "denominator": 3, "clean_successes": n_clean, "servo_successes": n_servo,
        "verified_policy_controls": 594, "verified_servo_decisions": 21,
        "servo_covers_clean_successes": covers,
        "no_observed_additional_wm_task_success_benefit_over_servo": no_extra_benefit,
        "interpretation": "Matched inputs, feedback schedule and physical budget; known three-seed development comparison. No population inference, blind confirmation, forecast-only causal attribution or attack/defense result.",
        "compute_boundary": "The hand-written controller uses far less compute and no learned forecast; this compares task competence, not compute-matched planning algorithms.",
        "model_calls": 0, "environment_steps": 0, "api_requests": 0, "training_updates": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out_dir / "paired_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print(json.dumps({"status": report["status"], "clean_successes": n_clean, "servo_successes": n_servo,
                      "no_extra_task_success_benefit": no_extra_benefit, "verified_controls": 594}))


if __name__ == "__main__":
    main()
