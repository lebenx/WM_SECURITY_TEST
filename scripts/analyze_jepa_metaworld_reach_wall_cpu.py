#!/usr/bin/env python3
"""Audit the fixed reach-wall CPU panel from archived real environment traces.

No runner/model/environment imports or execution. The shared physical trace
auditor is frozen; controller actions are independently reconstructed here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_dino_metaworld_clean_gate import bound, check, setup_arrays, sha, trace_outcome

BRANCHES = ["noop_repeat0", "noop_repeat1", "straight_servo", "wall_aware_servo"]
BOUNDARIES = [1, 16, 31, 46, 61, 76, 91]
HOLDS = [15] * 6 + [9]
TRACE_AUDITOR_SHA = "4aa118cf77e803d154beb03b976d55c187b82c3a3cbb589a11b2a4332d38dbfa"


def array_record(array):
    array = np.ascontiguousarray(array)
    metadata = {"dtype": array.dtype.str, "shape": list(array.shape)}
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return {**metadata, "sha256": hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()}


def audit_final(branch, rows):
    final = branch["final_observation"]
    with np.load(bound(final["npz"]), allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    check({k: array_record(v) for k, v in arrays.items()} == final["arrays"], "Final arrays changed")
    for key, record in rows[-1]["physical"].items():
        check(array_record(arrays[key]) == record, "Final physical array differs from trace")
    check(array_record(arrays["pixels"]) == rows[-1]["pixels"], "Final pixels differ from trace")
    check(arrays["state"].tolist() == rows[-1]["state"], "Final state differs from trace")
    check(np.array_equal(arrays["proprio"][-1], arrays["state"][:4]), "Final proprio differs from state")
    with Image.open(bound(final["image"])) as png:
        check(np.array_equal(np.asarray(png), arrays["pixels"][-1].transpose(1, 2, 0)), "Final PNG differs")


def audit_controller(branch, arrays, rows):
    decisions = json.loads(bound(branch["decisions"]).read_text())
    check(len(decisions) == 7, "Incomplete controller decisions")
    check([d["elapsed_before"] for d in decisions] == BOUNDARIES, "Feedback schedule changed")
    check([d["executed_control_count"] for d in decisions] == HOLDS, "Action holds changed")
    goal, current = arrays["goal_proprio"][-1], arrays["initial_proprio"][-1]
    for vector in (goal, current):
        check(vector.shape == (4,) and vector.dtype == np.float32 and np.isfinite(vector).all(),
              "Invalid raw proprio input")
    check(np.array_equal(current, arrays["initial_state"][:4]), "Initial proprio/state differ")
    offset, waypoint_queries = 0, 0
    for index, (decision, hold) in enumerate(zip(decisions, HOLDS), 1):
        if offset:
            current = np.asarray(rows[offset - 1]["state"][:4], dtype=np.float32)
            check(current.tolist() == rows[offset - 1]["state"][:4], "State was not exact float32 proprio")
        check(np.isfinite(current).all(), "Nonfinite actual policy input")
        desired = goal[:3].copy()
        x, y, z = current[:3]
        raised = (branch["condition"] == "wall_aware_servo"
                  and -.1 <= x <= .3 and .60 <= y <= .80 and z < .25)
        if raised:
            desired += np.asarray([0, 0, 1], dtype=np.float32)
            waypoint_queries += 1
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(np.float32(5) * (desired - current[:3]), -1, 1)
        check(decision["index"] == index and decision["current_proprio"] == current.tolist()
              and decision["goal_proprio"] == goal.tolist() and decision["action"] == action.tolist(),
              "Controller inputs/rule differ")
        for local, row in enumerate(rows[offset:offset + hold]):
            check(row["action"] == action.tolist() and row["planning_call"] == index
                  and row["action_in_chunk"] == local, "Actual control differs from registered hold")
        offset += hold
    check(offset == len(rows) == 99, "Incomplete trace")
    check(branch["decision_audit"] == {"all99_actions_from_recorded_raw_proprio": True,
                                       "decision_count": 7}, "Saved decision audit differs")
    return waypoint_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jepa_metaworld_reach_wall_cpu_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/jepa_metaworld_reach_wall_cpu_analysis_v1"))
    args = parser.parse_args()
    source = args.run_dir / "summary.json"
    saved = json.loads(source.read_text())
    config = json.loads(bound(saved["config"]).read_text())
    auditor = Path(__file__).with_name("analyze_dino_metaworld_clean_gate.py").resolve()
    check(sha(auditor) == TRACE_AUDITOR_SHA, "Frozen shared auditor changed")
    check(saved["status"] == "completed_cpu_validation", "CPU panel incomplete")
    check(saved["runner"] == config["runner"], "Runner differs")
    for key in ("runner", "frozen_helper", "servo_helper", "official_config", "model", "dino_encoder",
                "wall_policy_source", "resource_audit", "mujoco_installation"):
        bound(config[key])
    for name, digest in config["source_sha256"].items():
        check(sha(name) == digest, f"Registered source changed: {name}")
    check(saved["protocol"] == config["protocol"] and saved["gate"] == config["gate"], "Protocol differs")
    protocol = config["protocol"]
    check(protocol["task"] == "mw-reach-wall" and protocol["seeds"] == [1, 2, 3]
          and protocol["branch_order_per_seed"] == BRANCHES and protocol["query_elapsed_steps"] == BOUNDARIES
          and protocol["hold_control_counts"] == HOLDS, "Registered panel differs")
    check(saved["planned_branches"] == config["planned_branches"] ==
          [{"seed": s, "condition": c} for s in [1, 2, 3] for c in BRANCHES], "Planned branches differ")
    check(saved["model_forward_calls"] == saved["api_requests"] == saved["training_updates"] == 0
          and not saved["cuda_initialized"], "Unexpected model/GPU/API/training execution")
    loading = saved["strict_model_load"]
    check(loading["strict_loads"] and all(c["loaded_tensors"] > 0 and not c["missing"] and not c["unexpected"]
          for c in loading["strict_loads"]), "Strict loading incomplete")
    check(loading["all_parameters_float32"] and loading["model_action_dimension"] == 20
          and loading["model_forward_calls"] == 0, "Model load contract differs")
    check([e["seed"] for e in saved["episodes"]] == [1, 2, 3], "Seed panel incomplete or reordered")
    results, verified_controls, verified_decisions = [], 0, 0
    for episode in saved["episodes"]:
        check(episode["status"] == "completed" and episode["prerequisites_passed"], "Prerequisites invalid")
        check([b["condition"] for b in episode["branches"]] == BRANCHES, "Missing/changed branch")
        reference = episode["branches"][0]
        ref_arrays = setup_arrays(reference["setup"])
        for branch in episode["branches"]:
            check(branch["status"] == "completed", "Branch incomplete")
            arrays = setup_arrays(branch["setup"])
            check(arrays.keys() == ref_arrays.keys() and all(np.array_equal(arrays[k], ref_arrays[k]) for k in arrays),
                  "Different actual initial state, goal, or expert trace")
            for key in ("initial_image", "goal_image", "expert_trace"):
                check(branch["setup"][key]["sha256"] == reference["setup"][key]["sha256"], "Different paired artifact")
            outcome, rows = trace_outcome(branch["outcome"], branch["setup"], arrays,
                                          branch["condition"].startswith("noop"))
            audit_final(branch, rows)
            verified_controls += len(rows)
            waypoints = None
            if not branch["condition"].startswith("noop"):
                waypoints = audit_controller(branch, arrays, rows)
                verified_decisions += 7
            results.append({"seed": episode["seed"], "condition": branch["condition"],
                            "initial_reach_distance": branch["setup"]["initial_reach_distance"],
                            "expert_final_success": branch["setup"]["expert_final_success"],
                            "final_success": outcome["final_success"], "ever_success": outcome["ever_success"],
                            "final_reach_distance": outcome["final_reach_distance"],
                            "minimum_reach_distance": min(r["reach_distance"] for r in rows),
                            "reward_sum": outcome["reward_sum"], "raised_waypoint_queries": waypoints})
        check(reference["outcome"]["trace"]["sha256"] == episode["branches"][1]["outcome"]["trace"]["sha256"],
              "Noop repeats differ")
        check(reference["final_observation"]["arrays"] == episode["branches"][1]["final_observation"]["arrays"]
              and reference["final_observation"]["image"]["sha256"] == episode["branches"][1]["final_observation"]["image"]["sha256"],
              "Noop final observation repeats differ")
        check(all(all(group.values()) for group in episode["checks"].values()), "Saved pairing checks disagree")
    counts = {c: sum(r["final_success"] for r in results if r["condition"] == c)
              for c in ("noop_repeat0", "straight_servo", "wall_aware_servo")}
    gate = config["gate"]
    check(gate["minimum_native_final_successes"] == gate["minimum_native_minus_wall_aware_final_successes"]
          == gate["minimum_native_minus_noop_final_successes"] == 2, "Fixed future gate differs")
    stop = counts["noop_repeat0"] >= 2 or counts["wall_aware_servo"] >= 2
    decision = saved["decision"]
    check(decision["cpu_prerequisites_passed"] and decision["complete_panel"]
          and decision["final_success_counts"] == counts and decision["gpu_progression_allowed"] == (not stop),
          "CPU progression classification differs")
    check(decision["native_results_measured"] is False and decision["native_success_count"] is None
          and decision["native_gate_passed"] is None, "Native results fabricated from CPU controls")
    expected = ("stop_panel_native_unmeasured_mathematically_cannot_pass" if stop
                else "eligible_for_separately_registered_native_gpu_stage")
    check(decision["decision"] == expected, "Decision label differs")
    report = {"status": "completed_independent_reach_wall_cpu_audit",
              "source": {"path": str(source.resolve()), "sha256": sha(source)}, "config": saved["config"],
              "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
              "shared_auditor": {"path": str(auditor), "sha256": sha(auditor)},
              "source_files_checked": len(config["source_sha256"]), "strict_model_load": loading,
              "episodes": results, "denominator_per_condition": 3, "final_success_counts": counts,
              "verified_policy_controls": verified_controls, "verified_controller_decisions": verified_decisions,
              "decision": decision, "native_results_measured": False,
              "scope": "Three fixed development seeds; repeated controls are not independent samples. Wall prior is public; this is task screening, not learned forecast causality or security evidence.",
              "analysis_model_calls": 0, "analysis_environment_steps": 0, "api_requests": 0, "training_updates": 0}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out_dir / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({"status": report["status"], "final_success_counts": counts,
                      "verified_controls": verified_controls, "decision": expected}))


if __name__ == "__main__":
    main()
