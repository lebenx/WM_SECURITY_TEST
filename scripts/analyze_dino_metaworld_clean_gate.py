#!/usr/bin/env python3
"""Independently rebuild the small DINO/MetaWorld Clean gate from saved traces.

No runner imports, model loading, simulator steps, or API calls. Fixed seeds are
development examples, so this reports descriptive counts rather than a claimed
population confidence interval.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def bound(record):
    p = Path(record["path"])
    check(sha(p) == record["sha256"], f"File binding differs: {p}")
    return p


def close(a, b):
    check(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), f"Values differ: {a}, {b}")


def setup_arrays(setup):
    with np.load(bound(setup["setup"]), allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    check(set(arrays) == set(setup["arrays"]), "Setup fields differ")
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()
        check(digest == setup["arrays"][name]["sha256"], f"Setup array differs: {name}")
    for k in ("initial_image", "goal_image"):
        bound(setup[k])
    close(np.linalg.norm(arrays["tcp"] - arrays["target"]), setup["initial_reach_distance"])
    close(np.linalg.norm(arrays["expert_final_tcp"] - arrays["expert_target"]), setup["expert_final_reach_distance"])
    check(not setup["initial_success"] and setup["initial_reach_distance"] > .05, "Trivial start")
    check(setup["expert_final_success"] and setup["expert_final_reach_distance"] <= .05, "Unreachable expert goal")
    expert = [json.loads(line) for line in bound(setup["expert_trace"]).read_text().splitlines()]
    check(len(expert) == 100 and [e["step"] for e in expert] == list(range(1, 101)), "Expert budget differs")
    check(expert[-1]["success"] == setup["expert_final_success"], "Expert final success differs")
    check(np.array_equal(np.asarray([e["action"] for e in expert]), arrays["expert_actions"].reshape(-1, 4)), "Expert actions differ")
    check(np.array_equal(np.asarray(expert[-1]["state"]), arrays["goal_state"]), "Expert final/goal state differs")
    return arrays


def trace_outcome(saved, setup, arrays, is_noop):
    path = bound(saved["trace"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    check(len(rows) == 99, "Expected 99 controls after warmup")
    for i, row in enumerate(rows):
        check(row["environment_step"] == i + 2 and row["policy_step"] == i + 1, "Step numbering differs")
        check(row["done"] == (i == 98), "Unexpected termination")
        previous_tcp = arrays["tcp"] if i == 0 else np.asarray(rows[i - 1]["tcp_after"])
        check(np.array_equal(np.asarray(row["tcp_before"]), previous_tcp), "TCP trace continuity differs")
        distance = float(np.linalg.norm(np.asarray(row["tcp_after"]) - arrays["target"]))
        close(distance, row["reach_distance"])
        check(row["success"] == (distance <= .05), "Environment success disagrees with reach distance")
        if is_noop:
            check(row["action"] == [0., 0., 0., 0.], "No-op contains nonzero physical action")
    result = {
        "final_success": rows[-1]["success"],
        "ever_success": any(r["success"] for r in rows),
        "final_reach_distance": rows[-1]["reach_distance"],
        "total_environment_steps": rows[-1]["environment_step"],
        "policy_controls_executed": len(rows),
        "reward_sum": sum(r["reward"] for r in rows),
        "terminal_done": rows[-1]["done"],
    }
    for k, value in result.items():
        if isinstance(value, float):
            close(value, saved[k])
        else:
            check(value == saved[k], f"Outcome differs: {k}")
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("runs/dino_metaworld_clean_gate_v1"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/dino_metaworld_clean_gate_analysis_v1"))
    args = p.parse_args()
    source = args.run_dir / "summary.json"
    summary = json.loads(source.read_text())
    check(summary["status"] == "completed", "Incomplete run cannot pass the gate")
    config = json.loads(bound(summary["config"]).read_text())
    cpu = json.loads(bound(config["cpu_report"]).read_text())
    check(summary["runner"] == config["runner"] == cpu["runner"], "Runner identities differ")
    bound(summary["runner"])
    for key in ("model", "dino_weights", "official_config"):
        bound(config[key])
    for name, digest in config["source_sha256"].items():
        check(sha(name) == digest, f"Registered source changed: {name}")
    check(config["gate"] == summary["gate"]["definition"], "Gate definition changed")
    check(config["seeds"] == [1, 2, 3] == [e["seed"] for e in summary["episodes"]], "Seed panel changed")
    check(cpu["cpu_prerequisites_passed"], "CPU prerequisite failed")
    check(cpu["status"] == "completed_cpu_validation" and cpu["seeds"] == [1, 2, 3]
          and [e["seed"] for e in cpu["episodes"]] == [1, 2, 3], "CPU panel incomplete/mismatched")
    for name, digest in config["cpu_artifact_sha256"].items():
        check(sha(name) == digest, f"CPU evidence changed: {name}")
    results = []
    verified_controls = 0
    for clean, reference in zip(summary["episodes"], cpu["episodes"]):
        check(clean["status"] == "completed" and clean["seed"] == reference["seed"], "Episode incomplete/mismatched")
        initial = clean["setup"]
        arrays = setup_arrays(initial)
        clean_result, rows = trace_outcome(clean["outcome"], initial, arrays, False)
        verified_controls += len(rows)
        noop_results = []
        check(reference["status"] == "completed" and reference["all_exact"] and len(reference["repeats"]) == 2,
              "CPU paired repeats incomplete")
        for repeat in reference["repeats"]:
            other = repeat["setup"]
            other_arrays = setup_arrays(other)
            check(initial["arrays"] == other["arrays"], "Clean/no-op setup differs")
            for k in ("initial_image", "goal_image"):
                check(initial[k]["sha256"] == other[k]["sha256"], "Clean/no-op image differs")
            result, noop_rows = trace_outcome(repeat["noop"], other, other_arrays, True)
            verified_controls += len(noop_rows)
            noop_results.append(result)
        check(len(noop_results) == 2 and noop_results[0] == noop_results[1], "No-op repeats differ")
        check(reference["repeats"][0]["noop"]["trace"]["sha256"] == reference["repeats"][1]["noop"]["trace"]["sha256"],
              "No-op full trace bytes differ")
        check(clean["matched_noop"] == reference["repeats"][0]["noop"], "Wrong matched no-op reference")
        queries = clean["queries"]
        check([q["elapsed_before"] for q in queries] == [1, 16, 31, 46, 61, 76, 91], "Replanning schedule differs")
        check([q["planned_control_count"] for q in queries] == [15] * 6 + [10], "Planned chunk sizes differ")
        check([q["executed_control_count"] for q in queries] == [15] * 6 + [9], "Executed chunk sizes differ")
        actions = []
        for q in queries:
            check(len(q["minimum_cost_per_iteration"]) == 15, "CEM iterations differ")
            actions.extend(q["physical_actions"][:q["executed_control_count"]])
        check(actions == [r["action"] for r in rows], "Planned and executed controls differ")
        results.append({"seed": clean["seed"], "initial_reach_distance": initial["initial_reach_distance"],
                        "expert_goal_final_success": initial["expert_final_success"],
                        "clean_final_success": clean_result["final_success"], "noop_final_success": noop_results[0]["final_success"],
                        "clean_ever_success": clean_result["ever_success"], "noop_ever_success": noop_results[0]["ever_success"],
                        "clean_final_reach_distance": clean_result["final_reach_distance"],
                        "noop_final_reach_distance": noop_results[0]["final_reach_distance"],
                        "clean_policy_return": clean_result["reward_sum"], "noop_policy_return": noop_results[0]["reward_sum"]})
    check(len(results) == 3, "All three fixed pairs required")
    clean_count = sum(r["clean_final_success"] for r in results)
    noop_count = sum(r["noop_final_success"] for r in results)
    gate = clean_count >= config["gate"]["minimum_clean_final_successes"] and clean_count - noop_count >= config["gate"]["minimum_clean_minus_noop_final_successes"]
    check(gate == summary["gate"]["passed"], "Gate classification differs")
    check(clean_count == summary["gate"]["clean_final_successes"] and noop_count == summary["gate"]["noop_final_successes"], "Gate counts differ")
    report = {"status": "completed_independent_trace_audit", "gate_passed": gate,
              "source": {"path": str(source.resolve()), "sha256": sha(source)},
              "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
              "episodes": results, "clean_successes": clean_count, "noop_successes": noop_count, "denominator": len(results),
              "verified_policy_controls_including_noop_repeats": verified_controls,
              "clean_minus_noop_success_fraction": (clean_count - noop_count) / len(results),
              "interpretation": "Descriptive readiness gate on three fixed development seeds; not population inference, independent generalization, forecast-only causal attribution, attack success or defense evidence.",
              "expert_scope": "Separate expert trajectories generate the goal and show reachability, not a matched-policy baseline from the warmup state.",
              "api_requests": 0, "training_updates": 0, "model_calls": 0, "environment_steps": 0}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out_dir / "paired_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({"status": report["status"], "gate_passed": gate, "clean_successes": clean_count, "noop_successes": noop_count, "verified_controls": verified_controls}))


if __name__ == "__main__":
    main()
