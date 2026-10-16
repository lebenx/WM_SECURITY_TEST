#!/usr/bin/env python3
"""CPU-only audit of the fixed original-task N1/N2 development panel.

Uses saved environment outcomes, never model scores as task utility. No model,
environment, training, or CUDA initialization occurs here. The native producer
and its artifacts are read-only. Partial runs never yield a panel conclusion.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

import aggregate_cosmos_native_clean_gate as clean

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosmos_native_n1_vs_n2_matched30_v1.json"
CONFIG_SHA = "0b81e40fa3212678499e37ed96a3c5483702875e127493143c7c8f652c3dc891"
EXPECTED = {(t, i) for t in range(10) for i in range(3)}
require, sha, read_json = clean.require, clean.sha, clean.read_json


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row.get("task_id"), row.get("init_id")
        require(all(type(x) is int for x in key), "Non-integer task/init")
        require(key in EXPECTED and key not in indexed, f"Unexpected/duplicate pair: {key}")
        indexed[key] = row
    return indexed


def equal(a, b):
    return clean.exact_array(np.asarray(a), np.asarray(b))


def validate_clean_labels(record, goal):
    if "condition" in record:
        require(record["condition"] == "clean", "Non-clean record label")
    for field in ("policy_goal", "forecast_goal", "value_goal"):
        if field in record:
            require(record[field] == goal, f"Non-original Goal label: {field}")


def validate_inputs():
    cfg, config_sha = read_json(CONFIG)
    require(config_sha == CONFIG_SHA, "Preregistered protocol changed")
    for group in ("source_sha256", "reference_artifact_sha256", "physical_file_sha256"):
        for name, expected in cfg[group].items():
            require(sha(ROOT / name) == expected, f"Frozen {group} changed: {name}")
    require(set(index_rows(cfg["planned_episodes"])) == EXPECTED, "Registered panel changed")
    require(sum(r["known_N1_pair"] for r in cfg["planned_episodes"]) == 1, "Known panel changed")
    require(sum(r["new_N1_pair"] for r in cfg["planned_episodes"]) == 29, "New panel changed")
    n2_path = ROOT / cfg["n2_source_summary"]
    n2, n2_sha = read_json(n2_path)
    require(n2["status"] == "completed" and set(index_rows(n2["results"])) == EXPECTED,
            "N2 reference incomplete")
    audit, _ = read_json(ROOT / cfg["independent_n2_audit"])
    require(audit["source_sha256"] == n2_sha and audit["gate_passed"] is True
            and audit["complete_registered_grid"] is True and not audit["validation_errors"],
            "N2 independent validation stale/failed")
    gate, _ = read_json(clean.GATE)
    n2_audits = [clean.audit_episode(r, n2_path, gate) for r in n2["results"]]
    known_path = ROOT / cfg["known_n1_summary"]
    known, _ = read_json(known_path)
    require(known["status"] == "completed" and len(known["results"]) == 1, "Known N1 source changed")
    require(known["results"][0]["environment_steps"] == 146
            and known["results"][0]["success"] is True, "Known N1 outcome changed")
    known_audit = audit_n1(known["results"][0], known_path)
    return cfg, n2, n2_audits, known, known_audit


def audit_n1(row, summary_path):
    """Check native N1 records without fabricating a candidate/value record."""
    fixed = {"status": "completed", "condition": "clean", "suite": "libero_goal",
             "best_of_n": 1, "seed": 31, "environment_seed": 0, "max_steps": 300,
             "settle_steps": 10, "success_at_start": False}
    for k, v in fixed.items():
        require(type(row.get(k)) is type(v) and row[k] == v, f"N1 field mismatch: {k}")
    require(type(row.get("success")) is bool and type(row.get("terminal_done")) is bool,
            "N1 success/done must be Boolean")
    require(all(type(row.get(k)) is int for k in ("task_id", "init_id", "environment_steps", "action_chunks")),
            "N1 identifiers/counts must be integers")
    require((row["task_id"], row["init_id"]) in EXPECTED, "Unexpected N1 pair")
    steps = row["environment_steps"]
    require(0 < steps <= 300, "Invalid N1 steps")
    tp, cp = (clean.artifact(row[k], summary_path) for k in ("trace", "chunks"))
    trace, chunks = clean.records(tp), clean.records(cp)
    for record in [row] + chunks + trace:
        validate_clean_labels(record, row["goal"])
    require(len(trace) == steps and [r["step"] for r in trace] == list(range(1, steps + 1)),
            "Trace steps/count mismatch")
    require(len(chunks) == row["action_chunks"] == math.ceil(steps / 16), "Chunk count mismatch")
    for i, chunk in enumerate(chunks):
        first = i * 16 + 1
        require(chunk["chunk"] == i + 1 and chunk["first_step"] == first, "Chunk order changed")
        require(chunk["native_best_of_n"] is None, "N1 unexpectedly uses scored candidate selection")
        for k in ("normalized_actions", "environment_actions"):
            x = np.asarray(chunk[k], dtype=np.float64)
            require(x.shape == (16, 7) and np.isfinite(x).all(), f"Malformed N1 {k}")
        require(math.isfinite(chunk["future_slot_rms"]), "Nonfinite joint future diagnostic")
        for j, event in enumerate(trace[first - 1:first + 15]):
            require(event["chunk"] == i + 1 and equal(event["action"], chunk["environment_actions"][j]),
                    "Executed N1 control differs from joint policy proposal")
    for i, event in enumerate(trace):
        require(type(event["task_success"]) is bool and type(event["done"]) is bool,
                "Environment flags not Boolean")
        require(type(event["reward"]) in (float, int) and math.isfinite(event["reward"]), "Invalid reward")
        for k, shape in (("action", (7,)), ("eef_before", (3,)), ("eef_after", (3,)),
                         ("eef_quaternion", (4,)), ("gripper_qpos", (2,))):
            x = np.asarray(event[k], dtype=np.float64)
            require(x.shape == shape and np.isfinite(x).all(), f"Malformed trace {k}")
        if i:
            require(equal(trace[i - 1]["eef_after"], event["eef_before"]), "EEF discontinuity")
        if i < steps - 1:
            require(not event["task_success"] and not event["done"], "Continued past stopping event")
    require(row["success"] == trace[-1]["task_success"]
            and row["terminal_done"] == trace[-1]["done"], "Reported success/done differs from environment")
    require(row["success"] or row["terminal_done"] or steps == 300, "Unexplained early stop")
    require(row["reward_sum"] == sum(x["reward"] for x in trace)
            and row["final_reward"] == trace[-1]["reward"], "Reward totals disagree")
    require(row["action_norm_mean"] == float(np.mean([np.linalg.norm(x["action"]) for x in trace])),
            "Action norm differs from actual controls")
    sp = tp.with_name(tp.stem + "_states.npz")
    state_arrays = {}
    with np.load(sp, allow_pickle=False) as states:
        require(set(states.files) == {"initial", "settled", "final", "eef_path"}, "Saved state catalog changed")
        require(equal(states["eef_path"], np.asarray([e["eef_after"] for e in trace], dtype=np.float64)),
                "Saved EEF path disagrees with actual trace")
        for k in states.files:
            x = states[k]
            require(x.dtype.kind in "fiu" and np.isfinite(x).all(), "Invalid simulator array")
            state_arrays[k] = {"dtype": x.dtype.str, "shape": list(x.shape), "sha256": clean.array_sha(x)}
    paths = {"trace": tp, "chunks": cp, "states": sp,
             **{k: clean.artifact(row[k], summary_path) for k in ("initial_image", "final_image")},
             "episode_json": tp.with_suffix(".json")}
    episode, _ = read_json(paths["episode_json"])
    require(episode == row, "Episode JSON and summary row differ")
    return {"status": "valid", "task_id": row["task_id"], "init_id": row["init_id"],
            "success": row["success"], "environment_steps": steps,
            "steps_to_success": steps if row["success"] else None,
            "termination": "success" if row["success"] else "environment_done" if row["terminal_done"] else "step_limit",
            "artifacts": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
            "state_arrays": state_arrays}


def paired_prefix(n1, n2, n1_path, n2_path):
    for k in ("task_id", "init_id", "task_name", "goal", "initial_state_sha256", "bddl_sha256", "init_file_sha256"):
        require(n1[k] == n2[k], f"Physical/task pair mismatch: {k}")
    require(sha(clean.artifact(n1["initial_image"], n1_path)) == sha(clean.artifact(n2["initial_image"], n2_path)),
            "Paired initial image changed")
    t1, t2 = (clean.artifact(r["trace"], p) for r, p in ((n1, n1_path), (n2, n2_path)))
    with np.load(t1.with_name(t1.stem + "_states.npz"), allow_pickle=False) as a, \
            np.load(t2.with_name(t2.stem + "_states.npz"), allow_pickle=False) as b:
        for k in ("initial", "settled"):
            require(equal(a[k], b[k]), f"Paired simulator state changed: {k}")
    a, b = clean.records(t1), clean.records(t2)
    first = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if not equal(x["action"], y["action"])), None)
    prefix = min(len(a), len(b)) if first is None else first - 1
    require(a[:prefix] == b[:prefix], "Different physical trace despite identical controls")
    if first is None:
        require(len(a) == len(b), "Different stopping points without control divergence")
    ca, cb = (clean.records(clean.artifact(r["chunks"], p)) for r, p in ((n1, n1_path), (n2, n2_path)))
    matched = []
    for x, y in zip(ca, cb):
        if x["first_step"] > prefix + 1:
            break
        require(x["first_step"] == y["first_step"], "Query positions differ")
        candidate = y["native_best_of_n"]["candidates"][0]
        same = equal(x["environment_actions"], candidate["environment_actions"])
        normalized = (equal(x["normalized_actions"], y["normalized_actions"])
                      if y["native_best_of_n"]["selected_index"] == 0 else None)
        matched.append({"chunk": x["chunk"], "first_step": x["first_step"],
                        "n2_selected_index": y["native_best_of_n"]["selected_index"],
                        "candidate0_seed": candidate["seed"], "candidate0_environment_actions_exact": same,
                        "candidate0_normalized_actions_exact_when_archived": normalized})
    require(matched, "No common initial query")
    return {"initial_and_settled_states_exact": True, "initial_PNG_exact": True,
            "identical_physical_prefix_steps": prefix, "first_executed_action_difference_step": first,
            "matched_queries": matched,
            "candidate0_parity_passed": all(x["candidate0_environment_actions_exact"]
                and x["candidate0_seed"] == 31 and x["candidate0_normalized_actions_exact_when_archived"] is not False
                for x in matched),
            "scope": "Recorded identical physical prefix only, not post-divergence fixed candidates or a pure future ablation."}


def descriptive(rows):
    if not rows:
        return None
    both = [r for r in rows if r["n1_success"] and r["n2_success"]]
    return {"pairs": len(rows), "tasks": len({r["task_id"] for r in rows}),
            "n1_successes": sum(r["n1_success"] for r in rows),
            "n2_successes": sum(r["n2_success"] for r in rows),
            "success_difference_n2_minus_n1": float(np.mean([int(r["n2_success"]) - int(r["n1_success"]) for r in rows])),
            "n2_gains": sum(not r["n1_success"] and r["n2_success"] for r in rows),
            "n2_losses": sum(r["n1_success"] and not r["n2_success"] for r in rows),
            "mean_environment_steps_n1": float(np.mean([r["n1_steps"] for r in rows])),
            "mean_environment_steps_n2": float(np.mean([r["n2_steps"] for r in rows])),
            "mean_failure_penalized_cost_difference_n2_minus_n1": float(np.mean([r["n2_completion_cost"] - r["n1_completion_cost"] for r in rows])),
            "successful_both_pairs": len(both),
            "mean_successful_both_steps_difference_n2_minus_n1": float(np.mean([r["n2_steps"] - r["n1_steps"] for r in both])) if both else None,
            "scope": "Descriptive fixed development panel; successful-both step comparison is outcome-selected. No independent-task/population inference."}


def task_cluster_statistics(rows):
    require(len(rows) == 30 and set(index_rows(rows)) == EXPECTED,
            "Cluster inference requires all30 paired observations")
    rng = np.random.default_rng(314159)
    bootstrap_indices = rng.integers(0, 10, size=(10000, 10))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=10)), dtype=np.float64)
    results = {}
    for name, effect in (
        ("success_difference_n2_minus_n1", lambda r: int(r["n2_success"]) - int(r["n1_success"])),
        ("failure_penalized_cost_difference_n2_minus_n1", lambda r: r["n2_completion_cost"] - r["n1_completion_cost"]),
    ):
        effects = np.asarray([np.mean([effect(r) for r in rows if r["task_id"] == t])
                              for t in range(10)], dtype=np.float64)
        observed = float(effects.mean())
        bootstraps = effects[bootstrap_indices].mean(axis=1)
        null_absolute = np.abs((signs * effects[None, :]).mean(axis=1))
        results[name] = {
            "task_means_in_task_id_order": effects.tolist(),
            "task_weighted_mean": observed,
            "task_cluster_percentile_bootstrap_95pct_interval": np.quantile(bootstraps, [.025, .975]).tolist(),
            "task_level_exact_two_sided_sign_flip_p": float(np.mean(null_absolute >= abs(observed) - 1e-12)),
            "sign_patterns_enumerated": len(signs),
        }
    return {"scope": "Exploratory complete fixed development panel only; ten task clusters with three paired initializations each.",
            "bootstrap_resamples": 10000, "bootstrap_seed": 314159,
            "sign_flip_assumptions": "Independent task-level effects and sign symmetry under the null; exact refers to enumerating 1024 signs, not randomization of treatment or a population guarantee.",
            "limits": "Shared scenes/model and a chosen development panel may limit cluster independence/generalization; no multiplicity correction. The unbalanced new29 subset is descriptive only.",
            "metrics": results}


def write_csv_outputs(result, out):
    """Tables retain partial status; empty/invalid panels get no fabricated rows."""
    pairs = result.get("pairs", [])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pair_path = out.with_name("paired_results.csv")
    fields = ["analysis_status", "task_id", "init_id", "known_N1_pair", "n1_success", "n2_success",
              "n1_steps", "n2_steps", "n1_steps_to_success", "n2_steps_to_success",
              "n1_completion_cost", "n2_completion_cost", "success_difference_n2_minus_n1",
              "cost_difference_n2_minus_n1", "n1_reward_sum", "n2_reward_sum", "n1_termination",
              "n2_termination", "first_executed_action_difference_step", "candidate0_prefix_parity"]
    with pair_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in pairs:
            row = {k: r.get(k) for k in fields}
            row.update(analysis_status=result["status"],
                       success_difference_n2_minus_n1=int(r["n2_success"]) - int(r["n1_success"]),
                       cost_difference_n2_minus_n1=r["n2_completion_cost"] - r["n1_completion_cost"],
                       first_executed_action_difference_step=r["audit"]["first_executed_action_difference_step"],
                       candidate0_prefix_parity=r["audit"]["candidate0_parity_passed"])
            writer.writerow(row)
    task_path = out.with_name("task_results.csv")
    fields = ["analysis_status", "task_id", "completed_pairs", "registered_pairs", "n1_successes", "n2_successes",
              "n2_gains", "n2_losses", "success_difference_n2_minus_n1",
              "mean_failure_penalized_cost_difference_n2_minus_n1",
              "mean_environment_steps_n1", "mean_environment_steps_n2"]
    with task_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in range(10):
            stats = descriptive([r for r in pairs if r["task_id"] == task]) or {}
            writer.writerow({k: (result["status"] if k == "analysis_status" else task if k == "task_id"
                                  else stats.get("pairs", 0) if k == "completed_pairs"
                                  else 3 if k == "registered_pairs" else stats.get(k)) for k in fields})
    return {"paired": {"path": str(pair_path), "sha256": sha(pair_path)},
            "per_task": {"path": str(task_path), "sha256": sha(task_path)}}


def validate_run_provenance(source, cfg, n2):
    p = source["protocol"]
    for k, v in cfg["fixed_protocol"].items():
        require(type(p.get(k)) is type(v) and p[k] == v, f"N1 protocol changed: {k}")
    require(p.get("preflight") is False and source["condition"] == "clean", "Wrong run mode")
    planned = [{k: r[k] for k in ("task_id", "init_id", "goal")} for r in cfg["planned_episodes"]]
    require(source["planned_episodes"] == planned, "Planned panel/order changed")
    expected = copy.deepcopy(n2["provenance"])
    expected.update(sampler_seeds=[31], future_steps=None, value_steps=None,
                    native_ar_forecast_condition_slots=None, native_ar_value_condition_slots=None,
                    selection="single policy proposal")
    if source.get("status") == "loading" and not source["results"]:
        # The immutable producer records backend metadata only after loading.
        # This is pending execution, not a model/runtime mismatch.
        expected.pop("runtime_compatibility", None)
    require(source["provenance"] == expected, "N1 model/runtime provenance differs beyond prescribed N1 change")
    for r in source["results"]:
        require(r["provenance"] == source["provenance"], "Episode/runtime provenance differs")


def analyze(path):
    cfg, n2, n2_audits, known, _ = validate_inputs()
    source, source_sha = read_json(path)
    validate_run_provenance(source, cfg, n2)
    n1_rows, n2_rows = index_rows(source["results"]), index_rows(n2["results"])
    plans = index_rows(cfg["planned_episodes"])
    pairs, audits, errors, parity = [], [], [], None
    n2_path = ROOT / cfg["n2_source_summary"]
    for key, r in sorted(n1_rows.items()):
        try:
            require(r["goal"] == plans[key]["goal"], "Task Goal changed")
            audit = audit_n1(r, path)
            audits.append(audit)
            pair_audit = paired_prefix(r, n2_rows[key], path, n2_path)
            if not pair_audit["candidate0_parity_passed"]:
                errors.append(f"Candidate0 common-prefix parity failed: {key}")
            b = n2_rows[key]
            pairs.append({"task_id": key[0], "init_id": key[1], "known_N1_pair": key == (0, 0),
                          "n1_success": r["success"], "n2_success": b["success"],
                          "n1_steps": r["environment_steps"], "n2_steps": b["environment_steps"],
                          "n1_steps_to_success": r["environment_steps"] if r["success"] else None,
                          "n2_steps_to_success": b["environment_steps"] if b["success"] else None,
                          "n1_completion_cost": r["environment_steps"] if r["success"] else 300,
                          "n2_completion_cost": b["environment_steps"] if b["success"] else 300,
                          "n1_reward_sum": r["reward_sum"], "n2_reward_sum": b["reward_sum"],
                          "n1_termination": audit["termination"],
                          "n2_termination": "success" if b["success"] else "environment_done" if b["terminal_done"] else "step_limit",
                          "audit": pair_audit})
            if key == (0, 0):
                parity = clean.paired_compare(known["results"][0], r, ROOT / cfg["known_n1_summary"], path)
                parity["checks"]["all_joint_policy_chunk_fields"] = parity["checks"].pop(
                    "all_candidate_values_selections_and_actions")
                parity["scope"] = ("Complete N1 physical trace, joint policy normalized/environment controls, "
                                   "joint-future RMS, null candidate-selection records, simulator arrays and PNGs; "
                                   "exclude timing/memory/path/placement metadata, not full backend/latent parity.")
                parity["runtime_placement_difference"] = cfg["known_reference"]
                if not parity["all_exact"]:
                    errors.append("Known N1 Task0/init0 offload-to-resident scientific parity failed")
        except Exception as exc:
            errors.append(f"Pair {key}: {type(exc).__name__}: {exc}")
    complete = source["status"] == "completed" and set(n1_rows) == EXPECTED and len(pairs) == 30
    valid = not errors and parity is not None and parity["all_exact"]
    status = "completed" if complete and valid else "invalid" if errors else "incomplete"
    return {"status": status, "source_status": source["status"], "source_summary": str(path),
            "source_sha256": source_sha, "config_sha256": CONFIG_SHA, "analysis_script_sha256": sha(__file__),
            "reference_summary": str(n2_path), "reference_summary_sha256": sha(n2_path),
            "complete_registered_grid": complete, "valid": valid, "validation_errors": errors,
            "missing_pairs": [{"task_id": t, "init_id": i} for t, i in sorted(EXPECTED - set(n1_rows))],
            "known_N1_parity": parity, "n1_episode_audits": audits, "n2_episode_audits": n2_audits,
            "pairs": pairs, "available_pair_statistics": descriptive(pairs),
            "complete_panel_statistics": descriptive(pairs) if status == "completed" else None,
            "complete_panel_task_cluster_uncertainty": task_cluster_statistics(pairs) if status == "completed" else None,
            "new29_statistics": descriptive([r for r in pairs if not r["known_N1_pair"]]) if status == "completed" else None,
            "known1_statistics": descriptive([r for r in pairs if r["known_N1_pair"]]),
            "per_task_available": [{"task_id": t, "statistics": descriptive([r for r in pairs if r["task_id"] == t])} for t in range(10)],
            "contribution_verdict": "Describe the paired original-task N1/N2 differences; no new success threshold or attack claim." if status == "completed" else None,
            "causal_limits": cfg["causal_limits"], "uncertainty": cfg["analysis"]["uncertainty"],
            "training_updates": 0, "api_requests": 0, "gpu_used": False, "environment_constructed": False}


def preflight(out):
    cfg, n2, audits, known, known_audit = validate_inputs()
    log = ROOT / "runs/cosmos_native_n1_vs_n2_matched30_v1_cpu/native_preflight.log"
    text = log.read_text()
    start = text.index('{\n  "status": "cpu_preflight_passed"')
    native = json.loads(text[start:])
    require(len(native["episodes"]) == 30, "Native preflight panel changed")
    require(native["provenance"]["sampler_seeds"] == [31]
            and native["provenance"]["future_steps"] is None
            and native["provenance"]["value_steps"] is None, "Native preflight is not N1")
    reference = index_rows(n2["results"])[0, 0]
    prefix = paired_prefix(known["results"][0], reference, ROOT / cfg["known_n1_summary"], ROOT / cfg["n2_source_summary"])
    require(prefix["candidate0_parity_passed"], "Existing N1/N2 shared-prefix proposal mismatch")
    clean.write_json(out, {"status": "passed", "config_sha256": CONFIG_SHA,
        "analysis_script_sha256": sha(__file__), "native_preflight_log_sha256": sha(log),
        "registered_source_count": len(cfg["source_sha256"]),
        "registered_reference_count": len(cfg["reference_artifact_sha256"]),
        "registered_physical_file_count": len(cfg["physical_file_sha256"]),
        "n2_episodes_validated": len(audits), "known_n1_audit": known_audit,
        "existing_known_N1_vs_N2_common_prefix": prefix,
        "gpu_used": False, "environment_constructed": False, "model_forward_calls": 0})


def self_test():
    def rejects(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
    checks = {"duplicate_pairs_rejected": rejects(lambda: index_rows([{"task_id": 0, "init_id": 0}] * 2)),
              "boolean_task_rejected": rejects(lambda: index_rows([{"task_id": False, "init_id": 0}])),
              "different_controls_not_equal": not equal([0.], [1.]),
              "signed_zero_not_equal": not equal([0.], [-0.]),
              "empty_statistics_none": descriptive([]) is None,
              "nonclean_annotation_rejected": rejects(lambda: validate_clean_labels({"condition": "goal_swap"}, "original")),
              "wrong_goal_annotation_rejected": rejects(lambda: validate_clean_labels({"policy_goal": "wrong"}, "original")),
              "partial_panel_inference_rejected": rejects(lambda: task_cluster_statistics([]))}
    require(all(checks.values()), "CPU self-test failed")
    return {"status": "passed", "checks": checks, "analysis_script_sha256": sha(__file__), "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/cosmos_policy_native_task_gate_clean_n1_matched30_v1/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/cosmos_native_n1_vs_n2_matched30_analysis_v1/analysis.json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        clean.write_json(args.out, result)
    elif args.preflight:
        preflight(args.out)
        result, _ = read_json(args.out)
    else:
        try:
            result = analyze(args.summary.resolve())
        except Exception as exc:
            result = {"status": "invalid", "contribution_verdict": None,
                      "validation_errors": [f"{type(exc).__name__}: {exc}"],
                      "config_sha256": CONFIG_SHA, "analysis_script_sha256": sha(__file__), "gpu_used": False}
        result["csv_tables"] = write_csv_outputs(result, args.out)
        clean.write_json(args.out, result)
    print(json.dumps({k: result.get(k) for k in ("status", "complete_registered_grid", "contribution_verdict", "validation_errors")}, indent=2))
    print(str(args.out))


if __name__ == "__main__":
    main()
