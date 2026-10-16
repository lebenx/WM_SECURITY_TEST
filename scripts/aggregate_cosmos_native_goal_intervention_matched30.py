#!/usr/bin/env python3
"""Audit actual paired task outcomes; never infer harm from predicted value.

Task0 is an already inspected pilot. Tasks1..9 are the new extension. Task-level
resampling describes this fixed panel, not population generalization. No model,
environment, API, or training is used by this analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/summary.json"
SWAP = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1/summary.json"
OUT = ROOT / "runs/cosmos_native_goal_intervention_matched30_analysis_v1/analysis.json"
CONFIG = ROOT / "configs/cosmos_native_goal_intervention_matched30_v1.json"
REGISTERED_CONFIG_SHA = "b6985191a202fbc611a51f4bb9994c92d66f60ebc80ac5fe3a5f22d74e056e04"
EXPECTED = {(t, i) for t in range(10) for i in range(3)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_snapshot(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def equal_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(summary):
    indexed = {}
    for r in summary["results"]:
        require(type(r["task_id"]) is int and type(r["init_id"]) is int, "Task/init must be integers")
        key = (r["task_id"], r["init_id"])
        require(key in EXPECTED and key not in indexed, f"Unexpected/duplicate episode {key}")
        require(r["status"] == "completed", f"Noncompleted row {key}")
        indexed[key] = r
    return indexed


def audit_episode(r, condition, tasks):
    require(r["condition"] == condition, "Wrong condition")
    for k, v in {"suite": "libero_goal", "best_of_n": 2, "seed": 31,
                 "environment_seed": 0, "max_steps": 300, "settle_steps": 10,
                 "success_at_start": False}.items():
        require(type(r[k]) is type(v) and r[k] == v, f"Wrong episode setting {k}")
    require(type(r["success"]) is bool, "Success must be boolean")
    t, c = lines(r["trace"]), lines(r["chunks"])
    require(len(t) == r["environment_steps"] and 0 < len(t) <= 300, "Wrong trace length")
    require(len(c) == r["action_chunks"] == math.ceil(len(t) / 16), "Wrong chunk length")
    require([x["step"] for x in t] == list(range(1, len(t) + 1)), "Noncontiguous trace")
    require(all(x["task_success"] is False for x in t[:-1]), "Actions after success")
    require(all(x["done"] is False for x in t[:-1]), "Actions after termination")
    require(t[-1]["task_success"] is r["success"], "Summary/actual success disagreement")
    require(t[-1]["done"] is r["terminal_done"], "Done disagreement")
    if not r["success"]:
        require(len(t) == 300 or r["terminal_done"], "Unexplained early failure")
    for j, chunk in enumerate(c):
        first = j * 16 + 1
        require(chunk["chunk"] == j + 1 and chunk["first_step"] == first, "Chunk timing")
        choice = chunk["native_best_of_n"]
        candidates = choice["candidates"]
        require([x["index"] for x in candidates] == [0, 1], "Candidate index")
        require([x["seed"] for x in candidates] == [31, 32], "Candidate RNG")
        values = [x["predicted_value"] for x in candidates]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid value")
        require(choice["selected_index"] == max(range(2), key=lambda k: values[k]), "Not native argmax")
        for candidate in candidates:
            a = np.asarray(candidate["environment_actions"])
            require(a.shape == (16, 7) and np.isfinite(a).all(), "Invalid candidate")
        selected = candidates[choice["selected_index"]]["environment_actions"]
        require(equal_array(chunk["environment_actions"], selected), "Chunk/selected action mismatch")
        actual = t[first - 1:min(first + 15, len(t))]
        for k, row in enumerate(actual):
            require(row["chunk"] == j + 1 and equal_array(row["action"], selected[k]), "Executed different action")
        if condition == "goal_swap":
            for obj in [r, chunk]:
                require(obj["policy_goal"] == obj["value_goal"] == r["goal"] == tasks[r["task_id"]], "Policy/value goal changed")
                require(obj["forecast_goal"] == tasks[(r["task_id"] + 1) % 10], "Wrong cyclic future Goal")
                require(obj["condition"] == condition, "Intermediate condition label")
            require(all(x["condition"] == condition for x in actual), "Trace condition label")
    return t, c


def paired_prefix(clean, swap, tasks):
    cp, sp = clean["provenance"], swap["provenance"]
    for k in cp:
        if k not in {"runner_sha256", "goal_routing"}:
            require(k in sp and cp[k] == sp[k], f"Model/runtime provenance differs: {k}")
    require(sp["base_runner_sha256"] == cp["runner_sha256"], "Different underlying episode runner")
    for k in ["task_id", "init_id", "task_name", "goal", "initial_state_sha256", "bddl_sha256", "init_file_sha256"]:
        require(clean[k] == swap[k], f"Physical/task pairing changed {k}")
    require(sha(clean["initial_image"]) == sha(swap["initial_image"]), "Initial images differ")
    state_hashes = {}
    with np.load(Path(clean["trace"]).with_name(Path(clean["trace"]).stem + "_states.npz")) as a, np.load(Path(swap["trace"]).with_name(Path(swap["trace"]).stem + "_states.npz")) as b:
        for k in ["initial", "settled"]:
            require(equal_array(a[k], b[k]), f"Initial simulator state differs: {k}")
            state_hashes[k] = hashlib.sha256(a[k].tobytes()).hexdigest()
    ct, cc = audit_episode(clean, "clean", tasks)
    st, sc = audit_episode(swap, "goal_swap", tasks)
    first_action = next((i + 1 for i, (a, b) in enumerate(zip(ct, st)) if not equal_array(a["action"], b["action"])), None)
    prefix = min(len(ct), len(st)) if first_action is None else first_action - 1
    physical = ["step", "chunk", "action", "eef_before", "eef_after", "eef_quaternion", "gripper_qpos", "reward", "done", "task_success"]
    for a, b in zip(ct[:prefix], st[:prefix]):
        for k in physical:
            require(equal_array(a[k], b[k]), f"Physical trace differs before action intervention: step {a['step']} {k}")
    if first_action is None:
        require(len(ct) == len(st), "Different lengths without action divergence")
    matched = []
    for a, b in zip(cc, sc):
        if a["first_step"] > prefix + 1:
            break
        require(a["first_step"] == b["first_step"], "Misaligned candidate queries")
        x, y = a["native_best_of_n"], b["native_best_of_n"]
        for ca, cb in zip(x["candidates"], y["candidates"]):
            require(equal_array(ca["environment_actions"], cb["environment_actions"]), "Proposals changed at matched physical prefix")
        matched.append({"chunk": a["chunk"], "first_step": a["first_step"],
                        "clean_selected": x["selected_index"], "swap_selected": y["selected_index"],
                        "selection_flip": x["selected_index"] != y["selected_index"],
                        "candidate_actions_equal": True})
    require(bool(matched), "No matched first query")
    artifacts = {}
    for label, r in [("clean", clean), ("swap", swap)]:
        paths = {k: Path(r[k]) for k in ["trace", "chunks", "initial_image", "final_image"]}
        paths["states"] = Path(r["trace"]).with_name(Path(r["trace"]).stem + "_states.npz")
        artifacts[label] = {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}
    return {"paired_initial_state_byte_hashes": state_hashes, "paired_artifact_hashes": artifacts,
            "common_model_runtime_provenance_equal": True, "identical_physical_prefix_steps": prefix,
            "first_executed_action_difference_step": first_action,
            "matched_queries_only_before_state_divergence": matched,
            "matched_query_selection_flips": sum(x["selection_flip"] for x in matched),
            "scope": "Candidate equality is established only at queries before physical trajectories diverge; later closed-loop proposals may differ."}


def stats(rows, infer=False):
    if not rows:
        return None
    d = np.array([int(r["swap_success"]) - int(r["clean_success"]) for r in rows])
    tasks = sorted({r["task_id"] for r in rows})
    task_d = np.array([np.mean([int(r["swap_success"]) - int(r["clean_success"]) for r in rows if r["task_id"] == t]) for t in tasks])
    observed = abs(float(task_d.mean()))
    perm = [abs(float(np.mean(task_d * np.asarray(signs)))) for signs in itertools.product((-1, 1), repeat=len(tasks))]
    p = sum(x >= observed - 1e-12 for x in perm) / len(perm)
    rng = np.random.default_rng(314159)
    boot = task_d[rng.integers(0, len(tasks), size=(10000, len(tasks)))].mean(axis=1)
    losses, gains = int((d < 0).sum()), int((d > 0).sum())
    discordant = losses + gains
    mcnemar = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(losses, gains) + 1)) / 2 ** discordant) if discordant else 1.
    first_query_flips = [r["audit"]["matched_queries_only_before_state_divergence"][0]["selection_flip"] for r in rows if "audit" in r]
    prefix_flip_episodes = [r["audit"]["matched_query_selection_flips"] > 0 for r in rows if "audit" in r]
    return {"paired_episodes": len(rows), "tasks": len(tasks),
            "clean_successes": sum(r["clean_success"] for r in rows), "swap_successes": sum(r["swap_success"] for r in rows),
            "clean_success_rate": float(np.mean([r["clean_success"] for r in rows])),
            "swap_success_rate": float(np.mean([r["swap_success"] for r in rows])),
            "success_difference_swap_minus_clean": float(d.mean()),
            "task_weighted_success_difference_swap_minus_clean": float(task_d.mean()),
            "lost_clean_successes": losses, "gained_clean_failures": gains,
            "tasks_with_lost_success": len({r["task_id"] for r in rows if r["clean_success"] and not r["swap_success"]}),
            "initial_fixed_query_ranking_flip_count": sum(first_query_flips) if len(first_query_flips) == len(rows) else None,
            "initial_fixed_query_ranking_flip_rate": float(np.mean(first_query_flips)) if len(first_query_flips) == len(rows) else None,
            "episodes_with_any_matched_prefix_ranking_flip": sum(prefix_flip_episodes) if len(prefix_flip_episodes) == len(rows) else None,
            "ranking_scope": "Initial-query rate has one fixed physical state and two matched full action candidates per episode. Any-prefix flip is descriptive and stops at trajectory divergence; it is not an uncensored per-query flip rate.",
            "mean_capped_completion_steps_difference": float(np.mean([r["swap_capped_completion_steps"] - r["clean_capped_completion_steps"] for r in rows])),
            "task_cluster_bootstrap_95pct_interval": np.quantile(boot, [.025, .975]).tolist() if infer else None,
            "task_level_exact_two_sided_sign_flip_p": p if infer else None,
            "episode_level_exact_mcnemar_p_secondary": mcnemar if infer else None,
            "inference_enabled_only_after_complete_validated_panel": infer,
            "uncertainty_scope": "Exploratory fixed development panel. Bootstrap resamples tasks; sign-flip requires independent task effects with sign symmetry under the null (exact refers only to enumeration, not a randomized trial). Episode McNemar additionally assumes independent episodes, not established here. No population or independent generalization claim."}


def analyze(clean_path, swap_path):
    (clean, clean_sha), (swap, swap_sha) = read_snapshot(clean_path), read_snapshot(swap_path)
    require(clean["status"] == "completed", "Clean source incomplete")
    ci, si = index_rows(clean), index_rows(swap)
    require(set(ci) == EXPECTED, "Incomplete Clean panel")
    registration_audit = {"verified": False, "reason": "Legacy pilot subset only; cannot finalize new panel"}
    if "registration" in swap:
        config, config_sha = read_snapshot(CONFIG)
        require(config_sha == REGISTERED_CONFIG_SHA, "Registered protocol bytes changed")
        require(swap["registration"] == config and swap["protocol_sha256"] == config_sha,
                "Run/registered protocol mismatch")
        require(swap["protocol"] == config["fixed_protocol"], "Recorded runtime protocol differs from registration")
        require(swap["planned_episodes"] == config["planned_episodes"], "Registered grid changed")
        require(clean_path.resolve() == (ROOT / config["clean_source_summary"]).resolve(), "Unregistered Clean source")
        require(config["reference_artifact_sha256"][config["clean_source_summary"]] == clean_sha, "Registered Clean bytes changed")
        for rel, expected in config["source_sha256"].items():
            require(sha(ROOT / rel) == expected, f"Registered implementation changed: {rel}")
        for rel, expected in config["reference_artifact_sha256"].items():
            require(sha(ROOT / rel) == expected, f"Registered reference changed: {rel}")
        expected_runner = config["source_sha256"]["scripts/pilot_cosmos_native_goal_intervention_matched30.py"]
        for p in [swap["provenance"]] + [r["provenance"] for r in si.values()]:
            require(p["protocol_sha256"] == config_sha and p["runner_sha256"] == expected_runner,
                    "Unregistered runner/protocol provenance")
            require(p["clean_source_summary_sha256"] == clean_sha, "Stale Clean provenance")
        threshold = config["analysis_scope"]["mechanism_followup_threshold"]
        require(threshold["minimum_lost_clean_successes"] == 3 and threshold["minimum_distinct_tasks_with_loss"] == 2,
                "Follow-up criterion changed")
        reproduction = swap.get("clean_reproduction")
        fresh_audit = None
        if reproduction is not None:
            from aggregate_cosmos_native_clean_gate import paired_compare
            fresh = reproduction["result"]
            fresh_audit = paired_compare(ci[0, 0], fresh, clean_path, swap_path)
            require(fresh_audit["all_exact"] and reproduction["comparison"]["all_exact"], "Fresh Clean parity failed")
        require(not si or fresh_audit is not None, "Swap started before actual fresh Clean parity")
        registration_audit = {"verified": True, "config_sha256": config_sha,
                              "registered_source_count": len(config["source_sha256"]),
                              "registered_reference_count": len(config["reference_artifact_sha256"]),
                              "fresh_clean_independent_recheck": fresh_audit}
    tasks = {t: ci[t, 0]["goal"] for t in range(10)}
    rows, errors = [], []
    for key, sr in sorted(si.items()):
        cr = ci[key]
        try:
            audit = paired_prefix(cr, sr, tasks)
            rows.append({"task_id": key[0], "init_id": key[1], "known_pilot": key[0] == 0,
                         "clean_success": cr["success"], "swap_success": sr["success"],
                         "clean_steps": cr["environment_steps"], "swap_steps": sr["environment_steps"],
                         "clean_capped_completion_steps": cr["environment_steps"] if cr["success"] else 300,
                         "swap_capped_completion_steps": sr["environment_steps"] if sr["success"] else 300,
                         "audit": audit, "clean_trace_sha256": sha(cr["trace"]), "swap_trace_sha256": sha(sr["trace"])})
        except (ValueError, KeyError, OSError) as e:
            errors.append({"task_id": key[0], "init_id": key[1], "error": str(e)})
    complete = swap["status"] == "completed" and set(si) == EXPECTED and not errors and registration_audit["verified"]
    extension = [r for r in rows if r["task_id"] != 0]
    all_stats, new_stats = stats(rows, complete), stats(extension, complete)
    gate = None
    if complete:
        gate = new_stats["lost_clean_successes"] >= 3 and new_stats["tasks_with_lost_success"] >= 2
    return {"status": "validation_error" if errors else "completed" if complete else "incomplete",
            "source_status": swap["status"], "clean_source": str(clean_path), "swap_source": str(swap_path),
            "clean_source_sha256": clean_sha, "swap_source_sha256": swap_sha,
            "analysis_sha256": sha(__file__), "validated_pairs": len(rows), "expected_pairs": 30,
            "registration_audit": registration_audit,
            "missing_pairs": [list(x) for x in sorted(EXPECTED - set(si))], "validation_errors": errors,
            "all30_descriptive": all_stats, "new27_primary_extension": new_stats,
            "known_pilot3_replication": stats([r for r in rows if r["task_id"] == 0], complete),
            "candidate_mechanism_followup_gate": gate,
            "gate_scope": "At least 3 lost Clean successes in at least 2 new-extension tasks permits a subsequent mechanism/repair test only; not a claim of statistical confirmation, external exploitability, or novelty. No verdict before all30 complete.",
            "decision": None if not complete else "follow_up_mechanism_only" if gate else "NoGo_stop_fixed_cyclic_construction",
            "rows": rows, "forecast_error_measured_here": False, "defense_measured_here": False,
            "fixed_candidate_single_action_utility_measured_here": False,
            "utility_scope": "Actual complete native closed-loop task success and capped completion steps. These include continued intervention after the first selected action; not an isolated first-action utility with a fixed shared continuation.",
            "api_requests": 0, "new_model_calls": 0, "new_environment_steps": 0,
            "interpretation": "Internal future-query context intervention in a native forecast-driven planner. No universal Goal-invariance requirement or external attacker interface demonstrated."}


def self_test():
    require(stats([{"task_id": 1, "clean_success": True, "swap_success": True,
                    "clean_capped_completion_steps": 10, "swap_capped_completion_steps": 10}], True)["task_level_exact_two_sided_sign_flip_p"] == 1., "Null test")
    rows = [{"task_id": t, "clean_success": True, "swap_success": False,
             "clean_capped_completion_steps": 10, "swap_capped_completion_steps": 300} for t in range(1, 7)]
    s = stats(rows, True)
    require(s["task_level_exact_two_sided_sign_flip_p"] == .03125, "Task permutation enumeration")
    require(s["episode_level_exact_mcnemar_p_secondary"] == .03125, "Exact McNemar")
    require(s["task_cluster_bootstrap_95pct_interval"] == [-1., -1.], "Task bootstrap")
    require(not equal_array(np.array([0.]), np.array([-0.])), "Signed zero equality")
    require(not equal_array(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64)), "Dtype equality")
    try:
        index_rows({"results": [{"task_id": 1, "init_id": 0, "status": "completed"}] * 2})
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate episode accepted")
    print(json.dumps({"status": "passed", "checks": 7, "model_calls": 0, "environment_steps": 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-summary", type=Path, default=CLEAN)
    p.add_argument("--swap-summary", type=Path, default=SWAP)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    result = analyze(a.clean_summary, a.swap_summary)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.out)
    with (a.out.parent / "paired_results.csv").open("w", newline="") as f:
        fields = ["task_id", "init_id", "known_pilot", "clean_success", "swap_success", "clean_steps", "swap_steps",
                  "clean_capped_completion_steps", "swap_capped_completion_steps", "initial_query_ranking_flip",
                  "first_executed_action_difference_step"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            values = {k: row[k] for k in fields if k in row}
            values["initial_query_ranking_flip"] = row["audit"]["matched_queries_only_before_state_divergence"][0]["selection_flip"]
            values["first_executed_action_difference_step"] = row["audit"]["first_executed_action_difference_step"]
            writer.writerow(values)
    with (a.out.parent / "task_results.csv").open("w", newline="") as f:
        fields = ["analysis_status", "task_id", "paired_episodes", "clean_successes", "swap_successes",
                  "lost_clean_successes", "gained_clean_failures", "initial_fixed_query_ranking_flip_count",
                  "mean_capped_completion_steps_difference"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in sorted({r["task_id"] for r in result["rows"]}):
            s = stats([r for r in result["rows"] if r["task_id"] == task])
            writer.writerow({k: result["status"] if k == "analysis_status" else task if k == "task_id" else s[k] for k in fields})
    print(json.dumps({k: result[k] for k in ["status", "validated_pairs", "validation_errors", "decision"]}))


if __name__ == "__main__":
    main()
