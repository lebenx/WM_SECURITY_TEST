#!/usr/bin/env python3
"""Audit completed JEPA reach-wall Clean and frozen CPU controls offline.

No model, runner or simulator imports. The three fixed development seeds are
paired units; repeated controls and CEM queries never increase that denominator.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_dino_metaworld_clean_gate import bound, check, setup_arrays, sha, trace_outcome
from analyze_jepa_metaworld_reach_wall_cpu import (
    BRANCHES, BOUNDARIES, HOLDS, audit_controller, audit_final,
)

SEEDS = [1, 2, 3]
TRACE_SHA = "4aa118cf77e803d154beb03b976d55c187b82c3a3cbb589a11b2a4332d38dbfa"
CPU_AUDITOR_SHA = "431def765f2bce4f8026045070b6361deab129b167e9cd2fffe1d8ecc5db1bf4"


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify_nested_refs(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            bound(value)
        for child in value.values():
            verify_nested_refs(child)
    elif isinstance(value, list):
        for child in value:
            verify_nested_refs(child)


def setup_extra(setup, arrays, seed):
    check(setup["seed"] == seed and setup["episode_index"] == seed - 1
          and setup["elapsed_after_warmup"] == 1 and int(arrays["elapsed"]) == 1,
          "Setup seed or warmup changed")
    check(np.array_equal(arrays["initial_proprio"][-1], arrays["initial_state"][:4])
          and np.array_equal(arrays["goal_proprio"][-1], arrays["goal_state"][:4]),
          "Initial/goal proprio and state disagree")
    for prefix in ("initial", "goal"):
        with Image.open(bound(setup[prefix + "_image"])) as image:
            check(np.array_equal(np.asarray(image), arrays[prefix + "_pixels"][-1].transpose(1, 2, 0)),
                  "Initial/goal PNG differs from model input pixels")


def final_extra(branch, rows, initial):
    audit_final(branch, rows)
    with np.load(bound(branch["final_observation"]["npz"]), allow_pickle=False) as arrays:
        check(np.array_equal(arrays["tcp"], np.asarray(rows[-1]["tcp_after"])),
              "Final physical TCP differs from outcome TCP")
        check(np.array_equal(arrays["target"], initial["target"]) and int(arrays["elapsed"]) == 100,
              "Final target or elapsed controls changed")


def paired_setup(left, a, right, b):
    check(a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a),
          "Actual physical/observed/expert initial-goal arrays differ")
    for field in ("initial_image", "goal_image", "expert_trace"):
        check(left[field]["sha256"] == right[field]["sha256"], "Paired input/expert bytes differ")


def strict_load(record):
    expected = [("DinoVisionTransformer", 175), ("VisionTransformerAdaLN", 92),
                ("ProprioceptiveEmbedding", 2)]
    check([(v["class"], v["loaded_tensors"]) for v in record["strict_loads"]] == expected,
          "Unexpected loaded JEPA model modules")
    check(all(not v["missing"] and not v["unexpected"] for v in record["strict_loads"])
          and record["all_parameters_float32"] and record["model_action_dimension"] == 20,
          "Model load incomplete or precision/dimension differs")


def cpu_reaudit(cpu, config, previous):
    check(cpu["status"] == "completed_cpu_validation" and cpu["runner"] == config["runner"]
          and cpu["protocol"] == config["protocol"] and cpu["gate"] == config["gate"],
          "CPU status/configuration differs")
    check([e["seed"] for e in cpu["episodes"]] == SEEDS, "CPU panel incomplete/reordered")
    check(cpu["planned_branches"] == config["planned_branches"] ==
          [{"seed": s, "condition": c} for s in SEEDS for c in BRANCHES], "CPU panel schedule differs")
    check(cpu["model_forward_calls"] == cpu["api_requests"] == cpu["training_updates"] == 0
          and cpu["cuda_initialized"] is False, "CPU execution scope differs")
    strict_load(cpu["strict_model_load"])
    results, records, controls, decisions = [], {}, 0, 0
    for episode in cpu["episodes"]:
        seed = episode["seed"]
        check(episode["status"] == "completed" and episode["prerequisites_passed"]
              and [b["condition"] for b in episode["branches"]] == BRANCHES, "CPU branches incomplete")
        reference = episode["branches"][0]
        reference_arrays = setup_arrays(reference["setup"])
        for branch in episode["branches"]:
            check(branch["status"] == "completed", "CPU branch incomplete")
            arrays = setup_arrays(branch["setup"])
            setup_extra(branch["setup"], arrays, seed)
            paired_setup(reference["setup"], reference_arrays, branch["setup"], arrays)
            outcome, rows = trace_outcome(branch["outcome"], branch["setup"], arrays,
                                          branch["condition"].startswith("noop"))
            final_extra(branch, rows, arrays)
            controls += len(rows)
            waypoints = None
            if not branch["condition"].startswith("noop"):
                waypoints = audit_controller(branch, arrays, rows)
                decisions += 7
            results.append({"seed": seed, "condition": branch["condition"],
                            "initial_reach_distance": branch["setup"]["initial_reach_distance"],
                            "expert_final_success": branch["setup"]["expert_final_success"],
                            "final_success": outcome["final_success"], "ever_success": outcome["ever_success"],
                            "final_reach_distance": outcome["final_reach_distance"],
                            "minimum_reach_distance": min(r["reach_distance"] for r in rows),
                            "reward_sum": outcome["reward_sum"], "raised_waypoint_queries": waypoints})
            records[(seed, branch["condition"])] = (branch, arrays, outcome)
        repeat = episode["branches"][1]
        check(reference["outcome"]["trace"]["sha256"] == repeat["outcome"]["trace"]["sha256"]
              and reference["final_observation"]["arrays"] == repeat["final_observation"]["arrays"]
              and reference["final_observation"]["image"]["sha256"] == repeat["final_observation"]["image"]["sha256"],
              "Noop repeated trajectories/final observations differ")
    counts = {c: sum(r["final_success"] for r in results if r["condition"] == c)
              for c in ("noop_repeat0", "straight_servo", "wall_aware_servo")}
    check(previous["episodes"] == results and previous["denominator_per_condition"] == 3
          and previous["verified_policy_controls"] == controls == 1188
          and previous["verified_controller_decisions"] == decisions == 42
          and previous["final_success_counts"] == counts, "Independent CPU audit differs from fresh recomputation")
    decision = cpu["decision"]
    check(decision == previous["decision"] and decision["cpu_prerequisites_passed"]
          and decision["complete_panel"] and decision["final_success_counts"] == counts
          and counts["noop_repeat0"] < 2 and counts["wall_aware_servo"] < 2
          and decision["gpu_progression_allowed"] is True
          and decision["native_results_measured"] is False, "CPU progression prerequisites differ")
    return records, counts, controls, decisions


def native_queries(episode, rows, normalization):
    queries = episode["queries"]
    query_path = Path(episode["setup"]["setup"]["path"]).parent / "queries.json"
    check(json.loads(query_path.read_text()) == queries, "Saved query artifact differs from embedded queries")
    check(episode["pending_query"] is None and len(queries) == 7, "Pending/incomplete native query")
    check([q["index"] for q in queries] == list(range(1, 8))
          and [q["elapsed_before"] for q in queries] == BOUNDARIES
          and [q["executed_control_count"] for q in queries] == HOLDS
          and [q["planned_control_count"] for q in queries] == [15] * 6 + [10], "Native schedule differs")
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    check(mean.shape == std.shape == (4,) and np.isfinite(mean).all()
          and np.isfinite(std).all() and (std > 0).all(), "Invalid action normalization")
    offset = 0
    for query in queries:
        remaining_macro = max((101 - query["elapsed_before"]) // 5, 1)
        check(query["planning_steps_left"] == remaining_macro
              and query["horizon"] == min(6, remaining_macro), "Native horizon accounting differs")
        costs = np.asarray(query["minimum_cost_per_iteration"])
        check(costs.shape == (15, 1) and np.isfinite(costs).all(), "CEM iteration log incomplete/nonfinite")
        normalized = np.asarray(query["normalized_macro_actions"], dtype=np.float32)
        planned = query["planned_control_count"]
        check(normalized.shape == (planned // 5, 20) and np.isfinite(normalized).all(),
              "Macro-action shape/value differs")
        physical = normalized.reshape(-1, 4) * std + mean
        check(physical.tolist() == query["physical_actions"], "Native action denormalization differs")
        count = query["executed_control_count"]
        for local, row in enumerate(rows[offset:offset + count]):
            check(row["action"] == physical[local].tolist() and row["planning_call"] == query["index"]
                  and row["action_in_chunk"] == local, "Planned/executed action or chunk association differs")
        offset += count
    check(offset == len(rows) == 99, "Native execution trace incomplete")
    return binding(query_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jepa_metaworld_reach_wall_native_v1"))
    parser.add_argument("--cpu-analysis", type=Path,
                        default=Path("runs/jepa_metaworld_reach_wall_cpu_analysis_v1/analysis.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/jepa_metaworld_reach_wall_native_analysis_v1"))
    args = parser.parse_args()
    source = args.run_dir / "summary.json"
    saved = json.loads(source.read_text())
    check(saved["status"] == "completed", "Incomplete native run has no task gate result")
    trace_auditor = Path(__file__).with_name("analyze_dino_metaworld_clean_gate.py")
    cpu_auditor = Path(__file__).with_name("analyze_jepa_metaworld_reach_wall_cpu.py")
    check(sha(trace_auditor) == TRACE_SHA and sha(cpu_auditor) == CPU_AUDITOR_SHA, "Frozen independent auditor changed")
    config = json.loads(bound(saved["config"]).read_text())
    cpu = json.loads(bound(config["cpu_summary"]).read_text())
    cpu_config = json.loads(bound(config["cpu_config"]).read_text())
    previous = json.loads(args.cpu_analysis.read_text())
    for record in (config, saved, cpu, cpu_config, previous):
        verify_nested_refs(record)
    check(previous["status"] == "completed_independent_reach_wall_cpu_audit"
          and previous["source"] == config["cpu_summary"] == saved["cpu_summary"]
          and previous["config"] == config["cpu_config"] == cpu["config"]
          and previous["analysis_script"] == binding(cpu_auditor), "CPU audit/source linkage differs")
    check(saved["runner"] == config["runner"] and saved["protocol"] == config["protocol"]
          and config["cpu_protocol"] == cpu_config["protocol"] and config["gate"] == cpu_config["gate"],
          "Native runner/protocol/gate linkage differs")
    for record, fields in ((config, ("source_sha256", "cpu_artifact_sha256")),
                           (cpu_config, ("source_sha256",))):
        for field in fields:
            for path, digest in record[field].items():
                check(sha(path) == digest, "Registered source/artifact changed: " + path)
    protocol = config["protocol"]
    check(protocol["task"] == "mw-reach-wall" and config["seeds"] == protocol["seeds"] == SEEDS
          and protocol["query_elapsed_steps"] == BOUNDARIES and protocol["executed_control_counts"] == HOLDS
          and protocol["actual_image_size"] == 224 and protocol["predictor"] == "AdaLN"
          and protocol["image_normalization"] == [[.485, .456, .406], [.229, .224, .225]]
          and protocol["cem"] == {"samples": 300, "iterations": 15, "elites": 10, "horizon": 6,
                                  "num_act_stepped": 3, "alpha": .1, "context": 2},
          "Registered native model/planner protocol differs")
    check(saved["api_requests"] == saved["training_updates"] == 0
          and saved["cuda_runtime"]["allocator"] == "expandable_segments:True", "Native runtime scope differs")
    strict_load(saved["model_load"])
    check(saved["model_load"]["normalization"] == cpu["strict_model_load"]["normalization"],
          "CPU/native action normalization differs")
    records, counts, cpu_controls, controller_decisions = cpu_reaudit(cpu, cpu_config, previous)
    check([e["seed"] for e in saved["episodes"]] == SEEDS, "Native fixed-seed panel incomplete/reordered")
    results, query_files, native_controls = [], [], 0
    for episode in saved["episodes"]:
        seed = episode["seed"]
        check(episode["status"] == "completed", "Native episode incomplete")
        arrays = setup_arrays(episode["setup"])
        setup_extra(episode["setup"], arrays, seed)
        check(set(episode["matched_cpu_outcomes"]) == set(BRANCHES), "Wrong matched CPU branch set")
        for condition in BRANCHES:
            branch, other, _ = records[(seed, condition)]
            paired_setup(episode["setup"], arrays, branch["setup"], other)
            check(episode["matched_cpu_outcomes"][condition] == branch["outcome"], "Matched CPU outcome differs")
        check(episode["cpu_setup_parity"]["all_exact"] and
              set(episode["cpu_setup_parity"]["checks"]) == set(BRANCHES) and
              all(all(group.values()) for group in episode["cpu_setup_parity"]["checks"].values()),
              "Saved native/CPU parity differs")
        outcome, rows = trace_outcome(episode["outcome"], episode["setup"], arrays, False)
        final_extra(episode, rows, arrays)
        query_files.append(native_queries(episode, rows, saved["model_load"]["normalization"]))
        native_controls += len(rows)
        row = {"seed": seed, "initial_reach_distance": episode["setup"]["initial_reach_distance"],
               "native_final_success": outcome["final_success"], "native_ever_success": outcome["ever_success"],
               "native_final_reach_distance": outcome["final_reach_distance"],
               "native_minimum_reach_distance": min(r["reach_distance"] for r in rows),
               "native_policy_return": outcome["reward_sum"]}
        for condition in ("noop_repeat0", "straight_servo", "wall_aware_servo"):
            comparator = records[(seed, condition)][2]
            for metric in ("final_success", "ever_success", "final_reach_distance", "reward_sum"):
                row[condition + "_" + metric] = comparator[metric]
            row["native_minus_" + condition + "_distance"] = outcome["final_reach_distance"] - comparator["final_reach_distance"]
            row["native_minus_" + condition + "_return"] = outcome["reward_sum"] - comparator["reward_sum"]
        results.append(row)
    native_count = sum(r["native_final_success"] for r in results)
    definition = config["gate"]
    check(definition["minimum_native_final_successes"] == definition["minimum_native_minus_wall_aware_final_successes"]
          == definition["minimum_native_minus_noop_final_successes"] == 2, "Fixed gate threshold changed")
    passed = native_count >= 2 and native_count - counts["wall_aware_servo"] >= 2 and native_count - counts["noop_repeat0"] >= 2
    gate = saved["gate"]
    check(gate["definition"] == definition and gate["passed"] == passed
          and gate["native_final_successes"] == native_count and gate["cpu_final_successes"] == counts
          and gate["native_minus_comparator"] == {c: native_count - n for c, n in counts.items()}, "Native gate differs")
    comparisons = {}
    for condition in counts:
        differences = [int(r["native_final_success"]) - int(r[condition + "_final_success"]) for r in results]
        comparisons[condition] = {
            "native_success_wins": sum(v > 0 for v in differences),
            "native_success_losses": sum(v < 0 for v in differences),
            "success_ties": sum(v == 0 for v in differences),
            "native_minus_comparator_success_fraction": sum(differences) / 3,
            "mean_native_minus_comparator_distance": float(np.mean([r["native_minus_" + condition + "_distance"] for r in results])),
            "mean_native_minus_comparator_return": float(np.mean([r["native_minus_" + condition + "_return"] for r in results])),
        }
    report = {
        "status": "completed_independent_reach_wall_native_audit", "source": binding(source),
        "config": saved["config"], "cpu_summary": config["cpu_summary"], "cpu_analysis": binding(args.cpu_analysis),
        "analysis_script": binding(__file__), "frozen_cpu_auditor": binding(cpu_auditor),
        "frozen_physical_auditor": binding(trace_auditor), "query_files": query_files,
        "source_files_checked": len(config["source_sha256"]), "cpu_artifacts_checked": len(config["cpu_artifact_sha256"]),
        "episodes": results, "native_final_successes": native_count, "cpu_final_successes": counts,
        "denominator_per_condition": 3, "gate_passed": passed, "gate_definition": definition,
        "paired_descriptive_comparisons": comparisons,
        "verified_native_policy_controls": native_controls, "verified_cpu_policy_controls": cpu_controls,
        "verified_total_policy_controls": native_controls + cpu_controls,
        "verified_native_planning_queries": len(query_files) * 7,
        "verified_controller_decisions": controller_decisions,
        "scope": "Three fixed development seeds, not random population samples. No confidence intervals or significance claims. Two noop repeats verify reproducibility, not n=6. CEM queries and physical controls do not increase n. Real environment success, distance and returns are descriptive task utility; no forecast-only causal attribution, attack or defense evidence.",
        "comparator_scope": "Wall-aware P5 has a public geometry prior and goal endpoint proprio, with the same seven feedback boundaries/99 policy controls; it is not the full native expert or an optimal ordinary controller. Native CEM has much more computation and can vary all 15 controls in a chunk. Straight P5 is descriptive and not part of the predeclared gate.",
        "cost_scope": "Logged per-iteration minimum CEM candidate cost is not asserted to be the executed elite-mean action's cost or true utility.",
        "analysis_model_calls": 0, "analysis_environment_steps": 0, "api_requests": 0, "training_updates": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out_dir / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print(json.dumps({"status": report["status"], "gate_passed": passed,
                      "native_final_successes": native_count, "cpu_final_successes": counts,
                      "verified_total_controls": native_controls + cpu_controls}))


if __name__ == "__main__":
    main()
