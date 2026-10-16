#!/usr/bin/env python3
"""CPU-only audit of Clean Cosmos first-query candidate utility.

Both recorded candidates are physically executed from each fixed initial state,
with the same original-goal N2 continuation. This module reads only this normal
baseline and its Clean source. Uniform selection is an analytic expectation;
the two-candidate oracle is an unavailable hindsight bound. No models or
environments are imported, and existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = [(t, i) for t in range(10) for i in range(3)]
GRID = [(t, i, c) for t, i in PANEL for c in (0, 1)]
EXECUTION_FIELDS = {"query_seconds", "cuda_allocated_bytes"}
GOAL_FIELDS = ("policy_goal", "forecast_goal", "value_goal")
PROTOCOL = {"suite": "libero_goal", "task_ids": list(range(10)), "init_ids": [0, 1, 2],
            "seed": 31, "env_seed": 0, "best_of_n": 2, "offload": False,
            "max_steps": 300, "settle_steps": 10, "video_stride": 2, "cuda_memory_fraction": .28}


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def sha(value):
    h = hashlib.sha256()
    with resolve(value).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(value):
    return json.loads(resolve(value).read_text())


def records(value):
    return [json.loads(line) for line in resolve(value).read_text().splitlines() if line.strip()]


def file_record(value):
    return {"path": str(resolve(value)), "sha256": sha(value)}


def bound(value, expected):
    result = file_record(value)
    equal(result["sha256"], expected, f"Changed file: {value}")
    return result


def same_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype.str == b.dtype.str and a.shape == b.shape and a.tobytes() == b.tobytes()


def finite_array(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), label)
    return a


def audit_episode(ep, forced_index=None):
    expected = {"status": "completed", "condition": "clean", "suite": "libero_goal", "best_of_n": 2,
                "seed": 31, "environment_seed": 0, "max_steps": 300, "settle_steps": 10,
                "success_at_start": False}
    for key, value in expected.items():
        equal(ep.get(key), value, "Episode protocol mismatch: " + key)
    require(type(ep.get("task_id")) is int and type(ep.get("init_id")) is int
            and (ep["task_id"], ep["init_id"]) in PANEL, "Unexpected task/init")
    require(type(ep.get("success")) is bool and type(ep.get("terminal_done")) is bool, "Nonboolean outcome")
    for key in GOAL_FIELDS:
        if key in ep:
            equal(ep[key], ep["goal"], "Changed original goal: " + key)
    n = ep["environment_steps"]
    require(type(n) is int and 0 < n <= 300, "Invalid action budget")
    trace, chunks = records(ep["trace"]), records(ep["chunks"])
    equal(len(trace), n, "Physical trace length")
    equal(len(chunks), math.ceil(n / 16), "Chunk count")
    equal(ep["action_chunks"], len(chunks), "Recorded chunk count")
    for j, chunk in enumerate(chunks):
        equal(chunk["chunk"], j + 1, "Chunk index")
        equal(chunk["first_step"], j * 16 + 1, "Query timing")
        if "condition" in chunk:
            equal(chunk["condition"], "clean", "Nonclean model context")
        for key in GOAL_FIELDS:
            if key in chunk:
                equal(chunk[key], ep["goal"], "Changed chunk goal")
        actions = finite_array(chunk["environment_actions"], (16, 7), "Malformed actions")
        finite_array(chunk["normalized_actions"], (16, 7), "Malformed normalized actions")
        require(type(chunk["future_slot_rms"]) in (int, float) and math.isfinite(chunk["future_slot_rms"]), "Invalid future diagnostic")
        selection = chunk["native_best_of_n"]
        candidates = selection["candidates"]
        equal([c["index"] for c in candidates], [0, 1], "Candidate indices")
        equal([c["seed"] for c in candidates], [31, 32], "Candidate seeds")
        values = [c["predicted_value"] for c in candidates]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Invalid predicted values")
        selected = forced_index if j == 0 and forced_index is not None else max(range(2), key=lambda k: values[k])
        equal(selection["selected_index"], selected, "First forced index or subsequent native argmax differs")
        for c in candidates:
            finite_array(c["environment_actions"], (16, 7), "Malformed candidate")
        require(same_array(actions, np.asarray(candidates[selected]["environment_actions"], dtype=np.float64)), "Chosen controls do not match candidate")
        for k, event in enumerate(trace[j * 16:(j + 1) * 16]):
            equal(event["chunk"], j + 1, "Trace chunk index")
            require(same_array(np.asarray(event["action"], dtype=np.float64), actions[k]), "Executed control differs from committed candidate")
    for i, event in enumerate(trace):
        equal(event["step"], i + 1, "Trace action order")
        require(type(event["task_success"]) is bool and type(event["done"]) is bool, "Nonboolean trace outcome")
        require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"]), "Invalid physical reward")
        if "condition" in event:
            equal(event["condition"], "clean", "Nonclean trace")
        for key, shape in (("action", (7,)), ("eef_before", (3,)), ("eef_after", (3,)),
                           ("eef_quaternion", (4,)), ("gripper_qpos", (2,))):
            finite_array(event[key], shape, "Malformed physical field: " + key)
        require(i == n - 1 or not (event["task_success"] or event["done"]), "Actions after termination")
        if i:
            equal(event["eef_before"], trace[i - 1]["eef_after"], "Physical trace discontinuity")
    equal(ep["success"], trace[-1]["task_success"], "Success does not match environment")
    equal(ep["terminal_done"], trace[-1]["done"], "Done does not match environment")
    require(ep["success"] or ep["terminal_done"] or n == 300, "Incomplete nonterminal episode")
    equal(ep["reward_sum"], sum(r["reward"] for r in trace), "Reward sum")
    equal(ep["final_reward"], trace[-1]["reward"], "Final reward")
    norms = [np.linalg.norm(np.asarray(e["action"], dtype=np.float64)) for e in trace]
    equal(ep["action_norm_mean"], float(np.mean(norms)), "Action norm summary")
    state_path = resolve(ep["trace"]).with_name(resolve(ep["trace"]).stem + "_states.npz")
    with np.load(state_path, allow_pickle=False) as states:
        equal(set(states.files), {"initial", "settled", "final", "eef_path"}, "State array keys")
        require(same_array(states["eef_path"], np.asarray([e["eef_after"] for e in trace], dtype=np.float64)), "Stored EEF path")
        state_digests = {}
        for key in states.files:
            a = states[key]
            require(a.dtype.kind in "fiu" and np.isfinite(a).all(), "Nonfinite simulator state")
            state_digests[key] = {"dtype": a.dtype.str, "shape": list(a.shape), "bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    return {"task_id": ep["task_id"], "init_id": ep["init_id"], "forced_candidate_index": forced_index,
            "success": ep["success"], "executed_steps": n, "capped_steps": n if ep["success"] else 300,
            "artifacts": {**{k: file_record(ep[k]) for k in ("trace", "chunks", "initial_image", "final_image")},
                          "states": file_record(state_path)}, "state_arrays": state_digests}


def compare_with_source(actual, source, actual_audit, source_audit, selected_arm):
    for key in ("task_id", "init_id", "task_name", "goal", "initial_state_sha256", "bddl_sha256", "init_file_sha256"):
        equal(actual[key], source[key], "Source task/state identity: " + key)
    provenance = dict(actual["provenance"])
    require(provenance.pop("first_candidate_utility_wrapper", None) is not None, "Missing wrapper provenance")
    equal(provenance, source["provenance"], "Continuation numerical provenance")
    ac, sc = records(actual["chunks"]), records(source["chunks"])
    equal(ac[0]["native_best_of_n"]["candidates"], sc[0]["native_best_of_n"]["candidates"], "First full candidate actions/values changed")
    for left, right in zip(ac[0]["native_best_of_n"]["candidates"], sc[0]["native_best_of_n"]["candidates"]):
        for field in ("environment_actions", "predicted_value"):
            require(same_array(np.asarray(left[field], dtype=np.float64), np.asarray(right[field], dtype=np.float64)), "First candidate bytes changed")
    for key in ("initial", "settled"):
        equal(actual_audit["state_arrays"][key], source_audit["state_arrays"][key], "Initial/settled physical state changed")
    equal(sha(actual["initial_image"]), sha(source["initial_image"]), "Initial PNG bytes")
    if selected_arm:
        for key in ("success", "terminal_done", "environment_steps", "action_chunks", "reward_sum", "final_reward", "action_norm_mean"):
            equal(actual[key], source[key], "Selected-arm result differs: " + key)
        equal(sha(actual["trace"]), sha(source["trace"]), "Selected-arm whole physical trace bytes differ")
        def strip(chunks):
            return [{k: v for k, v in c.items() if k not in EXECUTION_FIELDS} for c in chunks]
        equal(json.dumps(strip(ac), sort_keys=True), json.dumps(strip(sc), sort_keys=True), "Selected-arm scientific query fields differ")
        equal(actual_audit["state_arrays"], source_audit["state_arrays"], "Selected-arm all archived state arrays differ")
        equal(sha(actual["final_image"]), sha(source["final_image"]), "Selected-arm final PNG bytes")
    return {"first_full_candidate_set_exact": True, "initial_and_settled_state_exact": True,
            "selected_arm_whole_trajectory_exact": True if selected_arm else None,
            "excluded_query_fields": sorted(EXECUTION_FIELDS),
            "physical_trace_fields_excluded": []}


def cluster_uncertainty(rows):
    """Task-cluster descriptions for the two prespecified native-vs-uniform endpoints."""
    task_values = {endpoint: np.array([np.mean([r[endpoint] for r in rows if r["task_id"] == t])
                                      for t in range(10)], dtype=np.float64)
                   for endpoint in ("native_minus_uniform_success", "native_minus_uniform_capped_steps")}
    rng = np.random.default_rng(314159)
    indices = rng.integers(0, 10, size=(10000, 10))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=10)))
    output = {}
    for endpoint, d in task_values.items():
        boot = d[indices].mean(axis=1)
        permuted = np.abs((signs * d[None]).mean(axis=1))
        output[endpoint] = {"task_mean_effects": d.tolist(), "mean": float(d.mean()),
            "task_bootstrap_95pct_interval": np.quantile(boot, [.025, .975]).tolist(),
            "task_exact_two_sided_sign_flip_p": float(np.mean(permuted >= abs(float(d.mean())) - 1e-12))}
    return {"tasks": 10, "initial_states_per_task": 3, "bootstrap_draws": 10000, "bootstrap_seed": 314159,
            "sign_assignments": 1024, "endpoints": output,
            "scope": "Descriptive fixed development panel. Task resampling assumes exchangeable task clusters; sign-flip requires independent task effects and null sign symmetry. Tasks share a scene/model and were not randomly sampled; exact enumeration is not a randomized-trial guarantee. No episode-independent or 60-trajectory sample-size inference."}


def summarize(rows, complete):
    if not complete:
        return {"status": "incomplete_no_contribution_verdict", "available_pairs": len(rows),
                "resource_gate": None, "task_cluster_uncertainty": None}
    equal(sorted((r["task_id"], r["init_id"]) for r in rows), PANEL, "Statistics require complete fixed panel")
    wins = sum(r["native_success"] > r["alternate_success"] for r in rows)
    losses = sum(r["native_success"] < r["alternate_success"] for r in rows)
    ties = 30 - wins - losses
    per_task = []
    for task in range(10):
        panel = [r for r in rows if r["task_id"] == task]
        per_task.append({"task_id": task, "pairs": 3,
            "native_minus_uniform_success": float(np.mean([r["native_minus_uniform_success"] for r in panel])),
            "native_minus_uniform_capped_steps": float(np.mean([r["native_minus_uniform_capped_steps"] for r in panel])),
            "native_vs_alternate_binary_wins": sum(r["native_success"] > r["alternate_success"] for r in panel),
            "native_vs_alternate_binary_losses": sum(r["native_success"] < r["alternate_success"] for r in panel)})
    positive_tasks = [r["task_id"] for r in per_task if r["native_minus_uniform_success"] > 0]
    policy_table = {}
    for policy in ("native", "fixed0", "fixed1", "uniform", "oracle"):
        policy_table[policy] = {"success_count_or_expectation": sum(r[policy + "_success"] for r in rows),
            "success_rate": float(np.mean([r[policy + "_success"] for r in rows])),
            "capped_steps_total_or_expectation": sum(r[policy + "_capped_steps"] for r in rows),
            "mean_capped_steps": float(np.mean([r[policy + "_capped_steps"] for r in rows])),
            "scope": ("Analytic 0.5/0.5 expectation from both measured rollouts; no additional random-policy episodes"
                       if policy == "uniform" else "Hindsight two-candidate success upper bound and capped-cost lower bound, not a deployable selector"
                       if policy == "oracle" else "Derived policy utility from physically measured candidate rollouts")}
    passed = wins - losses >= 3 and len(positive_tasks) >= 2
    return {"status": "complete_fixed_panel", "pairs": 30, "physical_candidate_episodes": 60,
            "policies": policy_table, "native_vs_alternate_binary_wins": wins,
            "native_vs_alternate_binary_losses": losses, "binary_ties_excluded_from_accuracy": ties,
            "pairwise_accuracy_on_binary_discordant_pairs": wins / (wins + losses) if wins + losses else None,
            "binary_oracle_regret_count": losses, "binary_oracle_regret_rate": losses / 30,
            "native_minus_uniform_success": (wins - losses) / 60,
            "native_minus_uniform_capped_steps": float(np.mean([r["native_minus_uniform_capped_steps"] for r in rows])),
            "resource_gate": {"passed": passed, "minimum_success_gain": .05, "minimum_net_positive_tasks": 2,
                "net_binary_wins": wins - losses, "positive_task_ids": positive_tasks,
                "meaning": "Resource allocation only, not statistical confirmation. Cost gains cannot rescue a failed success gate.",
                "decision": "eligible_for_separately_registered_normal_baseline_work" if passed else "stop_using_this_panel_as_strong_first_query_ranking_carrier"},
            "per_task": per_task, "task_cluster_uncertainty": cluster_uncertainty(rows)}


def make_pair(task, init, selected, candidates):
    a, b = candidates[0], candidates[1]
    native, alternate = candidates[selected], candidates[1 - selected]
    row = {"task_id": task, "init_id": init, "source_native_selected_index": selected,
           "candidate0_success": a["success"], "candidate1_success": b["success"],
           "candidate0_executed_steps": a["executed_steps"], "candidate1_executed_steps": b["executed_steps"],
           "candidate0_capped_steps": a["capped_steps"], "candidate1_capped_steps": b["capped_steps"],
           "native_success": int(native["success"]), "alternate_success": int(alternate["success"]),
           "native_capped_steps": native["capped_steps"], "alternate_capped_steps": alternate["capped_steps"],
           "fixed0_success": int(a["success"]), "fixed1_success": int(b["success"]),
           "fixed0_capped_steps": a["capped_steps"], "fixed1_capped_steps": b["capped_steps"],
           "uniform_success": .5 * (a["success"] + b["success"]),
           "uniform_capped_steps": .5 * (a["capped_steps"] + b["capped_steps"]),
           "oracle_success": int(a["success"] or b["success"]),
           "oracle_capped_steps": min(a["capped_steps"], b["capped_steps"])}
    row["native_minus_uniform_success"] = row["native_success"] - row["uniform_success"]
    row["native_minus_uniform_capped_steps"] = row["native_capped_steps"] - row["uniform_capped_steps"]
    return row


def load_registration(filename):
    config = read(filename)
    equal(config.get("scope"), "clean_native_first_query_ranking_contribution", "Registration scope")
    equal(config.get("schema_version"), 1, "Registration schema")
    equal(config.get("status"), "prospective_clean_only_registration", "Registration status")
    for key, value in PROTOCOL.items():
        equal(config["protocol"].get(key), value, "Registered protocol: " + key)
    for key, value in {"candidate_indices": [0, 1], "candidate_seeds": [31, 32],
                       "camera_size": 256, "chunk_length": 16}.items():
        equal(config["protocol"].get(key), value, "Registered candidate protocol: " + key)
    for key, value in {"minimum_native_minus_alternate_net_successes": 3,
                       "minimum_native_minus_uniform_success_rate": .05,
                       "minimum_tasks_with_positive_native_minus_alternate_net_successes": 2}.items():
        equal(config["resource_gate"].get(key), value, "Registered resource gate: " + key)
    for key in ("api_requests", "training_updates"):
        equal(config[key], 0, "Unexpected API or training")
    for category in ("source_sha256", "reference_sha256", "physical_file_sha256", "model_sha256"):
        require(bool(config[category]), "Empty artifact inventory")
        for name, expected in config[category].items():
            bound(name, expected)
    bound(config["runner"]["path"], config["runner"]["sha256"])
    equal(config["source_sha256"].get(str(Path(__file__).relative_to(ROOT))), sha(__file__), "Analysis code is not frozen in registration")
    source_info = config["source_summary"]
    equal(resolve(source_info["path"]), ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/summary.json", "Fixed source panel")
    equal(source_info["sha256"], "075fe874619b45e9f1dde03ceafcfee238063e06f97ad104bd9cd97f93b0b50a", "Known source summary")
    bound(source_info["path"], source_info["sha256"])
    source = read(source_info["path"])
    equal(source["status"], "completed", "Source completeness")
    equal(source["condition"], "clean", "Source must use original goal")
    for key, value in PROTOCOL.items():
        equal(source["protocol"][key], value, "Source protocol: " + key)
    rows, audits, selected, plan = {}, {}, {}, []
    for ep in source["results"]:
        key = ep["task_id"], ep["init_id"]
        require(key not in rows, "Duplicate source pair")
        equal(ep["provenance"], source["provenance"], "Source per-episode provenance")
        check = audit_episode(ep)
        rows[key], audits[key] = ep, check
        first = records(ep["chunks"])[0]["native_best_of_n"]
        require(first["candidates"][0]["predicted_value"] != first["candidates"][1]["predicted_value"], "Unexpected first-query value tie")
        selected[key] = first["selected_index"]
        ep_path = resolve(ep["trace"]).with_suffix(".json")
        equal(read(ep_path), ep, "Source episode JSON/summary mismatch")
        for p in [ep_path] + [resolve(r["path"]) for r in check["artifacts"].values()]:
            equal(config["reference_sha256"].get(str(p.relative_to(ROOT))), sha(p), "Unbound source scientific artifact")
        bddl = resolve(ep["bddl"])
        equal(config["physical_file_sha256"].get(str(bddl.relative_to(ROOT))), ep["bddl_sha256"], "Unbound BDDL")
        init = ROOT / "vendor/LIBERO/libero/libero/init_files" / bddl.parent.name / (bddl.stem + ".pruned_init")
        equal(config["physical_file_sha256"].get(str(init.relative_to(ROOT))), ep["init_file_sha256"], "Unbound init file")
    equal(sorted(rows), PANEL, "Complete Clean source grid")
    for key in PANEL:
        s = selected[key]
        for c in (s, 1 - s):
            plan.append({"task_id": key[0], "init_id": key[1], "forced_candidate_index": c,
                         "source_native_selected_index": s, "role": "native_selected" if c == s else "alternate"})
    equal(config["planned_branches"], plan, "Fixed selected-then-alternate branch plan")
    return config, source, rows, audits, selected


def analyze(registration, summary_path):
    config, source, sources, source_audits, selected = load_registration(registration)
    result = {"status": "incomplete", "registration": file_record(registration), "analysis_code": file_record(__file__),
              "source_summary": config["source_summary"], "complete_registered_grid": False,
              "validated_branches": 0, "expected_branches": 60, "episode_audits": [], "pairs": [],
              "contribution": summarize([], False), "gpu_used": False, "environment_created": False,
              "scope": "First-query candidate utility with the original-goal N2 continuation, not every-query ranking or forecast-only causal contribution.",
              "known_outcomes": "All30 original-selected outcomes were known; alternate outcomes are newly measured. Development panel, not a blind or randomly sampled evaluation.",
              "limits": ["One 16-control candidate block changes; total budget is 300 including that block.",
                         "Subsequent policy rule is shared, while observations and actions may differ normally between branches.",
                         "The two rollouts from an initial state are paired potential outcomes, not independent samples.",
                         "Task-cluster uncertainty is descriptive; tasks share a scene and checkpoint.",
                         "This normal baseline neither activates nor revises earlier failed scientific gates."]}
    if not resolve(summary_path).exists():
        result["reason"] = "No summary yet; no utilities imputed"
        return result
    summary_bytes = resolve(summary_path).read_bytes()
    doc = json.loads(summary_bytes)
    result["summary"] = {"path": str(resolve(summary_path)),
                         "sha256": hashlib.sha256(summary_bytes).hexdigest()}
    result["runner_status"] = doc.get("status")
    equal(resolve(summary_path).parent, resolve(config["out_dir"]), "Registered output directory")
    equal(doc["config"], {"path": str(resolve(registration)), "sha256": sha(registration)}, "Frozen registration binding")
    equal(doc["runner_sha256"], config["runner"]["sha256"], "Frozen runner binding")
    for key in ("source_summary", "protocol", "resource_gate", "planned_branches", "scope"):
        equal(doc[key], config[key], "Run/registration mismatch: " + key)
    equal(doc["condition"], "first_candidate_utility", "Run condition")
    equal(doc["api_requests"], 0, "No API calls")
    equal(doc["training_updates"], 0, "No training")
    provenance = dict(doc["provenance"])
    wrapper = provenance.pop("first_candidate_utility_wrapper")
    equal(provenance, source["provenance"], "Original continuation provenance changed")
    equal(wrapper["config"], doc["config"], "Wrapper config binding")
    equal(wrapper["runner_sha256"], doc["runner_sha256"], "Wrapper source binding")
    equal(wrapper["scope"], config["scope"], "Wrapper scope")
    observed = {}
    require(len(doc["results"]) <= 60, "Too many result branches")
    for index, outer in enumerate(doc["results"]):
        planned = config["planned_branches"][index]
        for key, value in planned.items():
            equal(outer.get(key), value, "Unregistered branch/order: " + key)
        equal(outer["condition"], "first_candidate_utility", "Truthful outer condition")
        t, i, c = outer["task_id"], outer["init_id"], outer["forced_candidate_index"]
        ep = outer["episode"]
        equal(ep["provenance"], doc["provenance"], "Episode numerical provenance")
        equal((ep["task_id"], ep["init_id"]), (t, i), "Nested task/init IDs")
        bound(outer["episode_json"]["path"], outer["episode_json"]["sha256"])
        equal(read(outer["episode_json"]["path"]), ep, "Nested episode JSON differs")
        bound(outer["source_episode_json"]["path"], outer["source_episode_json"]["sha256"])
        equal(read(outer["source_episode_json"]["path"]), sources[t, i], "Wrong source episode")
        fq = outer["first_query_parity"]
        bound(fq["path"], fq["sha256"])
        proof = read(fq["path"])
        equal(fq["all_exact"], True, "Recorded query parity failed")
        equal(proof["all_exact"], True, "First-query parity file failed")
        require(bool(proof["checks"]) and all(v is True for v in proof["checks"].values()), "Invalid first-query check record")
        equal(proof["forced_candidate_index"], c, "First-query forced-index proof")
        equal(proof["source_native_selected_index"], selected[t, i], "First-query native selection proof")
        equal(proof["source_chunks"], file_record(sources[t, i]["chunks"]), "First-query source chunk binding")
        equal(outer["initial_parity"]["all_exact"], True, "Initial parity failed")
        require(all(v is True for v in outer["initial_parity"]["checks"].values()), "Initial parity checks")
        for item in outer["artifacts"].values():
            bound(item["path"], item["sha256"])
        checked = audit_episode(ep, forced_index=c)
        native_arm = c == selected[t, i]
        checked["independent_source_comparison"] = compare_with_source(ep, sources[t, i], checked, source_audits[t, i], native_arm)
        if native_arm:
            equal(outer["selected_arm_parity"]["all_exact"], True, "Recorded selected-arm parity failed")
            equal(outer["selected_arm_parity"]["trace_file_bytes_exact"], True, "Recorded trace-byte parity failed")
        else:
            equal(outer["selected_arm_parity"], None, "Alternate branch mislabeled as selected parity")
        observed[t, i, c] = checked
        result["episode_audits"].append(checked)
    result["validated_branches"] = len(observed)
    result["missing_branches"] = [{"task_id": t, "init_id": i, "forced_candidate_index": c} for t, i, c in GRID if (t, i, c) not in observed]
    for t, i in PANEL:
        if all((t, i, c) in observed for c in (0, 1)):
            result["pairs"].append(make_pair(t, i, selected[t, i], {c: observed[t, i, c] for c in (0, 1)}))
    complete = doc.get("status") == "completed" and not result["missing_branches"]
    result["complete_registered_grid"] = complete
    if doc.get("status") == "completed":
        require(complete, "Terminal run lacks registered branches")
    result["status"] = "completed_validated" if complete else "incomplete"
    result["contribution"] = summarize(result["pairs"], complete)
    return result


def self_test():
    def sample(win_keys):
        rows = []
        for t, i in PANEL:
            candidates = {0: {"success": True, "executed_steps": 100, "capped_steps": 100},
                          1: {"success": (t, i) not in win_keys,
                              "executed_steps": 100 if (t, i) not in win_keys else 300,
                              "capped_steps": 100 if (t, i) not in win_keys else 300}}
            rows.append(make_pair(t, i, 0, candidates))
        return rows
    rows = sample({(0, 0), (0, 1), (1, 0)})
    output = summarize(rows, True)
    require(output["resource_gate"]["passed"] and output["native_minus_uniform_success"] == .05, "Net-three/two-task boundary")
    require(output["pairwise_accuracy_on_binary_discordant_pairs"] == 1 and output["binary_ties_excluded_from_accuracy"] == 27, "Binary tie exclusion")
    require(output["policies"]["uniform"]["success_count_or_expectation"] == 28.5, "Uniform is an expectation")
    require(not summarize(sample({(0, 0), (0, 1), (0, 2)}), True)["resource_gate"]["passed"], "One task cannot pass")
    ties = sample(set())
    for row in ties:
        row["native_minus_uniform_capped_steps"] = -100
    require(not summarize(ties, True)["resource_gate"]["passed"], "Cost cannot rescue success gate")
    require(summarize(rows[:29], False)["resource_gate"] is None, "Partial has no verdict")
    require(summarize(sample(set()), True)["pairwise_accuracy_on_binary_discordant_pairs"] is None, "All ties have undefined accuracy")
    require("torch" not in sys.modules and "libero" not in sys.modules, "CPU-only import boundary")
    return {"status": "passed", "scope": "Synthetic arithmetic only, no reported experimental outcomes",
            "checks": ["threshold_boundary", "positive_task_count", "binary_ties", "uniform_expectation", "cost_cannot_rescue", "partial_no_verdict", "CPU_only"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if not all((args.registration, args.summary, args.out_dir)):
        parser.error("--registration, --summary and --out-dir are required")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in ("analysis.json", "paired_results.csv", "task_results.csv", "policy_results.csv")]
    if any(p.exists() for p in paths):
        parser.error("Existing output evidence; use a new directory")
    try:
        result = analyze(args.registration, args.summary)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "invalid", "error": {"type": type(error).__name__, "message": str(error)},
                  "analysis_code": file_record(__file__), "pairs": [], "contribution": None,
                  "resource_gate": None, "gpu_used": False, "environment_created": False}
    with paths[0].open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    contribution = result.get("contribution") or {}
    table_rows = [result["pairs"], contribution.get("per_task", []),
                  [{"policy": k, **v} for k, v in contribution.get("policies", {}).items()]]
    for filename, rows in zip(paths[1:], table_rows):
        with filename.open("x", newline="") as handle:
            fields = list(rows[0]) if rows else ["status"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": result["status"], "analysis": str(paths[0]),
                      "resource_gate": contribution.get("resource_gate"), "error": result.get("error")}))
    sys.exit(0 if result["status"] == "completed_validated" else 2)


if __name__ == "__main__":
    main()
