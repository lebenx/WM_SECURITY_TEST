#!/usr/bin/env python3
"""CPU-only, post-hoc initial-query context contrasts from completed paired logs.

Requires all 30 real episodes and the completed independent pair audit. Reuses
already generated scores; never queries a model, simulator, or utility target.
The two mixed contexts are separate fixed diagnostics, not evaluated policies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/summary.json"
SWAP = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1/summary.json"
PAIR_AUDIT = ROOT / "runs/cosmos_native_goal_intervention_matched30_analysis_v1/analysis.json"
PAIR_SCRIPT = ROOT / "scripts/aggregate_cosmos_native_goal_intervention_matched30.py"
CONFIG = ROOT / "configs/cosmos_native_goal_intervention_matched30_v1.json"
OUT = ROOT / "runs/cosmos_candidate_context_contrasts_v1/analysis.json"
GRID = {(t, i) for t in range(10) for i in range(3)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(data), sha_bytes(data)


def array_sha(value):
    a = np.asarray(value)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return sha_bytes(header + b"\0" + np.ascontiguousarray(a).tobytes())


def index_rows(rows):
    result = {}
    for row in rows:
        require(type(row.get("task_id")) is int and type(row.get("init_id")) is int,
                "Task/init identifiers must be actual integers")
        key = row["task_id"], row["init_id"]
        require(key in GRID and key not in result, "Duplicate or unexpected task/init pair")
        result[key] = row
    require(set(result) == GRID, "Exactly all 30 registered task/init pairs are required")
    return result


def validate_complete_gate(clean, swap, audit, clean_sha, swap_sha):
    require(clean.get("status") == swap.get("status") == "completed",
            "Both source summaries must be completed; no partial diagnostic output")
    ci, si, ai = (index_rows(x["results"] if "results" in x else x["rows"])
                  for x in (clean, swap, audit))
    require(all(r.get("status") == "completed" and r.get("condition") == "clean" for r in ci.values()),
            "All Clean rows must be completed Clean episodes")
    require(all(r.get("status") == "completed" and r.get("condition") == "goal_swap" for r in si.values()),
            "All Swap rows must be completed GoalSwap episodes")
    require(audit.get("status") == audit.get("source_status") == "completed"
            and audit.get("validated_pairs") == audit.get("expected_pairs") == 30
            and audit.get("missing_pairs") == [] and audit.get("validation_errors") == [],
            "Independent pair audit must have completed and validated the entire panel")
    require(audit.get("registration_audit", {}).get("verified") is True,
            "Independent audit did not verify protocol/source registration")
    require(audit.get("clean_source_sha256") == clean_sha
            and audit.get("swap_source_sha256") == swap_sha, "Independent audit is stale")
    for row in ai.values():
        initial = row["audit"]["matched_queries_only_before_state_divergence"][0]
        require(initial["chunk"] == 1 and initial["first_step"] == 1
                and initial["candidate_actions_equal"] is True,
                "Independent audit did not establish matched initial candidates")
        require(row["audit"].get("common_model_runtime_provenance_equal") is True,
                "Independent audit did not establish shared model/runtime provenance")
    return ci, si, ai


def values_and_candidates(chunk):
    require(chunk["chunk"] == 1 and chunk["first_step"] == 1, "Only the first query is permitted")
    selection = chunk["native_best_of_n"]
    candidates = selection["candidates"]
    require([r["index"] for r in candidates] == [0, 1]
            and [r["seed"] for r in candidates] == [31, 32], "Candidate order/seeds changed")
    values = np.asarray([r["predicted_value"] for r in candidates], dtype=np.float64)
    require(values.shape == (2,) and np.isfinite(values).all()
            and ((values >= 0) & (values <= 1)).all(), "Malformed native values")
    require(type(selection["selected_index"]) is int
            and selection["selected_index"] == int(np.argmax(values)), "Non-native selection/tie break")
    actions = [np.asarray(r["environment_actions"], dtype=np.float64) for r in candidates]
    require(all(a.shape == (16, 7) and np.isfinite(a).all() for a in actions), "Malformed action controls")
    return values, [array_sha(a) for a in actions]


def contrast(clean_values, swap_values):
    c, s = np.asarray(clean_values, dtype=np.float64), np.asarray(swap_values, dtype=np.float64)
    require(c.shape == s.shape == (2,) and np.isfinite(c).all() and np.isfinite(s).all(), "Invalid contrast")
    delta = s - c
    common = float(delta.mean())
    differential = float(delta[1] - delta[0])
    clean_margin, swap_margin = float(c[1] - c[0]), float(s[1] - s[0])
    arms = {"clean_C0_C1": c, "uniform_swap_S0_S1": s,
            "swap_candidate0_S0_C1": np.array([s[0], c[1]]),
            "swap_candidate1_C0_S1": np.array([c[0], s[1]])}
    clean_selected = int(np.argmax(c))
    energy = float(np.square(delta).sum())
    return {
        "C0_C1": c.tolist(), "S0_S1": s.tolist(), "candidate_score_changes": delta.tolist(),
        "common_mode_shift": common, "differential_margin_change": differential,
        "clean_margin_C1_minus_C0": clean_margin, "swap_margin_S1_minus_S0": swap_margin,
        "abs_common_over_abs_clean_margin": abs(common) / abs(clean_margin) if clean_margin else None,
        "clean_exact_tie": clean_margin == 0,
        "abs_common_exceeds_abs_clean_margin": abs(common) > abs(clean_margin),
        "candidate_changes_same_nonzero_sign": bool(delta[0] * delta[1] > 0),
        "common_mode_fraction_of_score_shift_energy": 2 * common**2 / energy if energy else None,
        "arms": {name: {"scores": score.tolist(), "selected_index": int(np.argmax(score)),
                        "margin_candidate1_minus_candidate0": float(score[1] - score[0]),
                        "ranking_flip_vs_clean": int(np.argmax(score)) != clean_selected}
                 for name, score in arms.items()},
    }


def summarize(rows):
    require(bool(rows), "Empty descriptive group")
    ratios = [r["abs_common_over_abs_clean_margin"] for r in rows
              if r["abs_common_over_abs_clean_margin"] is not None]
    arms = ("uniform_swap_S0_S1", "swap_candidate0_S0_C1", "swap_candidate1_C0_S1")
    return {"initial_queries": len(rows), "tasks": len({r["task_id"] for r in rows}),
            "ranking_flips_vs_clean": {a: sum(r["arms"][a]["ranking_flip_vs_clean"] for r in rows) for a in arms},
            "ranking_flip_rates_vs_clean": {a: float(np.mean([r["arms"][a]["ranking_flip_vs_clean"] for r in rows])) for a in arms},
            "mean_abs_common_mode_shift": float(np.mean([abs(r["common_mode_shift"]) for r in rows])),
            "mean_abs_differential_margin_change": float(np.mean([abs(r["differential_margin_change"]) for r in rows])),
            "mean_abs_clean_margin": float(np.mean([abs(r["clean_margin_C1_minus_C0"]) for r in rows])),
            "median_abs_common_over_abs_clean_margin_excluding_exact_ties": float(np.median(ratios)) if ratios else None,
            "exact_clean_ties": sum(r["clean_exact_tie"] for r in rows),
            "common_mode_exceeds_clean_margin_count": sum(r["abs_common_exceeds_abs_clean_margin"] for r in rows),
            "same_nonzero_sign_candidate_shift_count": sum(r["candidate_changes_same_nonzero_sign"] for r in rows),
            "scope": "Descriptive initial-query scores only; no task utility, confidence interval, attacker choice, or confirmatory inference."}


def analyze(clean_path, swap_path, audit_path):
    clean, clean_sha = read_json(clean_path)
    swap, swap_sha = read_json(swap_path)
    audit, audit_sha = read_json(audit_path)
    ci, si, ai = validate_complete_gate(clean, swap, audit, clean_sha, swap_sha)
    config, config_sha = read_json(CONFIG)
    require(audit["analysis_sha256"] == sha_bytes(PAIR_SCRIPT.read_bytes()), "Pair-audit implementation changed")
    require(audit["registration_audit"]["config_sha256"] == config_sha
            and swap["protocol_sha256"] == config_sha and swap["registration"] == config,
            "Protocol registration differs from the completed independent audit")
    require(config["reference_artifact_sha256"][config["clean_source_summary"]] == clean_sha,
            "Clean baseline differs from frozen registration")
    require(swap["provenance"]["runner_sha256"] == config["source_sha256"]["scripts/pilot_cosmos_native_goal_intervention_matched30.py"],
            "Different intervention implementation")
    rows = []
    for key in sorted(GRID):
        audited = ai[key]["audit"]["paired_artifact_hashes"]
        chunks = {}
        for arm, episode in (("clean", ci[key]), ("swap", si[key])):
            for name, item in audited[arm].items():
                data = Path(item["path"]).read_bytes()
                require(sha_bytes(data) == item["sha256"], f"Post-audit artifact changed: {arm}/{name}/{key}")
                if name == "chunks":
                    require(Path(item["path"]).resolve() == Path(episode["chunks"]).resolve(), "Wrong chunk source")
                    chunks[arm] = json.loads(data.splitlines()[0])
        c, ca = values_and_candidates(chunks["clean"])
        s, sa = values_and_candidates(chunks["swap"])
        require(ca == sa, "Initial candidate controls changed between contexts")
        goal, swap_goal = config["goals_by_task_id"][key[0]], config["goals_by_task_id"][(key[0]+1) % 10]
        for field in ("policy_goal", "value_goal"):
            require(chunks["swap"][field] == goal, "Original policy/value Goal changed")
        require(chunks["swap"]["forecast_goal"] == swap_goal, "Nonregistered forecast Goal")
        first_audit = ai[key]["audit"]["matched_queries_only_before_state_divergence"][0]
        require(first_audit["clean_selected"] == int(np.argmax(c))
                and first_audit["swap_selected"] == int(np.argmax(s)), "Pair audit and score selection disagree")
        rows.append({"task_id": key[0], "init_id": key[1], "known_pilot": key[0] == 0,
                     "original_goal": goal, "cyclic_future_goal": swap_goal,
                     "candidate_control_sha256": ca,
                     "clean_chunks_sha256": audited["clean"]["chunks"]["sha256"],
                     "swap_chunks_sha256": audited["swap"]["chunks"]["sha256"],
                     **contrast(c, s)})
    return {"status": "completed_descriptive_score_diagnostic", "script_sha256": sha_bytes(Path(__file__).read_bytes()),
            "inputs": {"clean": {"path": str(clean_path), "sha256": clean_sha},
                       "uniform_swap": {"path": str(swap_path), "sha256": swap_sha},
                       "independent_pair_audit": {"path": str(audit_path), "sha256": audit_sha},
                       "registration": {"path": str(CONFIG), "sha256": config_sha}},
            "definitions": {"delta_i": "S_i-C_i", "common_mode": "(delta_0+delta_1)/2",
                            "differential_margin_change": "delta_1-delta_0",
                            "clean_margin": "C1-C0", "native_tie_break": "first candidate (index0)",
                            "fixed_mixed_forms": ["[S0,C1]", "[C0,S1]"]},
            "all30_descriptive": summarize(rows),
            "new27_relative_to_original_uniform_pilot": summarize([r for r in rows if r["task_id"] != 0]),
            "known_pilot3": summarize([r for r in rows if r["task_id"] == 0]),
            "by_task": {str(t): summarize([r for r in rows if r["task_id"] == t]) for t in range(10)},
            "rows": rows,
            "scope_limits": ["This diagnostic was proposed after inspecting partial uniform-intervention scores; all30 and new27 are descriptive, not prospective confirmation of mixed-context effects.",
                             "Mixed arms only combine previously generated first-query scores. No mixed-context policy or task was executed.",
                             "Both fixed mixed forms are reported separately. No condition, Goal, candidate, or seed is selected using task outcomes.",
                             "No task-success, reward, completion-time or physical-target values enter these contrasts or their summaries.",
                             "A ranking flip is not evidence of reduced actual utility. Existing full-episode uniform outcomes are not counterfactual utilities for mixed selections.",
                             "Goal-dependent conditional forecasts are not labeled wrong. Common decision information/target integrity is distinct from universal Goal-invariant forecasting.",
                             "No externally reachable candidate-specific Goal permission, new defense, or novelty claim is established."],
            "task_utility_inferred": False, "forecast_error_measured": False,
            "mixed_context_policy_executed": False, "defense_tested": False,
            "new_model_calls": 0, "new_environment_steps": 0, "api_requests": 0, "gpu_used": False}


def self_test():
    checks = {}
    c = contrast([.6, .7], [.4, .5])
    checks["common_shift_cancels_uniform_but_mixed_can_flip"] = (
        not c["arms"]["uniform_swap_S0_S1"]["ranking_flip_vs_clean"]
        and c["arms"]["swap_candidate1_C0_S1"]["ranking_flip_vs_clean"]
        and not c["arms"]["swap_candidate0_S0_C1"]["ranking_flip_vs_clean"])
    checks["zero_shift_leaves_all_arms_unchanged"] = not any(
        a["ranking_flip_vs_clean"] for a in contrast([.2, .8], [.2, .8])["arms"].values())
    tie = contrast([.5, .5], [.5, .5])
    checks["exact_ties_use_native_first_and_no_infinite_ratio"] = (
        tie["abs_common_over_abs_clean_margin"] is None and all(a["selected_index"] == 0 for a in tie["arms"].values()))
    clean = {"status": "completed", "results": [dict(task_id=t, init_id=i, status="completed", condition="clean") for t, i in sorted(GRID)]}
    swap = {"status": "completed", "results": [dict(task_id=t, init_id=i, status="completed", condition="goal_swap") for t, i in sorted(GRID)]}
    audit = {"status": "completed", "source_status": "completed", "validated_pairs": 30, "expected_pairs": 30,
             "missing_pairs": [], "validation_errors": [], "registration_audit": {"verified": True},
             "clean_source_sha256": "clean", "swap_source_sha256": "swap",
             "rows": [dict(task_id=t, init_id=i, audit={"common_model_runtime_provenance_equal": True,
                      "matched_queries_only_before_state_divergence": [{"chunk": 1, "first_step": 1, "candidate_actions_equal": True}]}) for t, i in sorted(GRID)]}
    validate_complete_gate(clean, swap, audit, "clean", "swap")
    checks["complete_registered_panel_accepted"] = True
    mutations = [lambda c, s, a: s.update(status="running"),
                 lambda c, s, a: s["results"].pop(),
                 lambda c, s, a: s["results"].append(s["results"][0]),
                 lambda c, s, a: a.update(status="incomplete"),
                 lambda c, s, a: a.update(swap_source_sha256="stale"),
                 lambda c, s, a: a["rows"][0]["audit"]["matched_queries_only_before_state_divergence"][0].update(candidate_actions_equal=False),
                 lambda c, s, a: a["registration_audit"].update(verified=False)]
    for i, mutate in enumerate(mutations):
        args = [copy.deepcopy(x) for x in (clean, swap, audit)]
        mutate(*args)
        try:
            validate_complete_gate(*args, "clean", "swap")
        except ValueError:
            checks[f"invalid_input_{i}_rejected"] = True
        else:
            checks[f"invalid_input_{i}_rejected"] = False
    require(all(checks.values()), "CPU contract failure")
    return {"status": "passed", "checks": checks, "count": len(checks),
            "script_sha256": sha_bytes(Path(__file__).read_bytes()), "gpu_used": False,
            "model_calls": 0, "environment_steps": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-summary", type=Path, default=CLEAN)
    p.add_argument("--swap-summary", type=Path, default=SWAP)
    p.add_argument("--pair-audit", type=Path, default=PAIR_AUDIT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return
    require(not a.out.exists(), "Preserve an existing frozen diagnostic")
    result = analyze(a.clean_summary.resolve(), a.swap_summary.resolve(), a.pair_audit.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(a.out), "sha256": sha_bytes(a.out.read_bytes()),
                      "all30_descriptive": result["all30_descriptive"],
                      "new27": result["new27_relative_to_original_uniform_pilot"]}, allow_nan=False))


if __name__ == "__main__":
    main()
