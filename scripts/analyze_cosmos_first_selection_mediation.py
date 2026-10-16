#!/usr/bin/env python3
"""Join existing Cosmos first selections to measured original-Goal continuations.

Standard-library, read-only analysis of existing experiment caches. No model,
simulator, runner import, new trajectory, or file hashing is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWAP = ROOT / "runs/cosmos_native_goal_intervention_matched30_analysis_v1/analysis.json"
UTILITY = ROOT / "runs/cosmos_first_candidate_utility_recovery_analysis_v1/analysis.json"
PRIOR_AUDIT = ROOT / "reports/cosmos_first_candidate_utility_recovery_independent_check_20261014.json"
SCENE = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1_scene_audit.json"
CONFIG = ROOT / "configs/cosmos_native_goal_intervention_matched30_v1.json"
OUT = ROOT / "reports/cosmos_first_selection_mediation_20261015.json"
PHYSICAL = ("step", "chunk", "action", "eef_before", "eef_after", "eef_quaternion",
            "gripper_qpos", "reward", "done", "task_success")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def first_jsonl(path):
    with Path(path).open() as handle:
        return json.loads(next(handle))


def episode(trace_path):
    """Read real stopping outcomes and the first block; reuse the prior full audit."""
    trace_path = Path(trace_path)
    doc = read(trace_path.with_suffix(".json"))
    require(doc["status"] == "completed", f"Incomplete source: {trace_path}")
    require(doc["success_at_start"] is False, "Already successful initial state")
    block, count, ever, last = [], 0, False, None
    with trace_path.open() as handle:
        for line in handle:
            last = json.loads(line)
            count += 1
            ever = ever or last["task_success"]
            if count <= 16:
                block.append({key: last[key] for key in PHYSICAL})
    require(0 < count <= 300 and count == doc["environment_steps"], "Real trace length mismatch")
    require(type(ever) is bool and ever == last["task_success"] == doc["success"],
            "Real trace success disagrees with summary/stopping")
    chunk_path = Path(doc["chunks"])
    chunk = first_jsonl(chunk_path)
    require(chunk["chunk"] == chunk["first_step"] == 1, "Not a first query")
    selection = chunk["native_best_of_n"]
    candidates = selection["candidates"]
    require([c["index"] for c in candidates] == [0, 1]
            and [c["seed"] for c in candidates] == [31, 32], "Candidate identity mismatch")
    for candidate in candidates:
        actions = candidate["environment_actions"]
        require(len(actions) == 16 and all(len(a) == 7 for a in actions), "Wrong action shape")
        require(all(math.isfinite(v) for a in actions for v in a), "Nonfinite action")
    selected = selection["selected_index"]
    require(type(selected) is int and selected in (0, 1), "Invalid selection")
    executed = candidates[selected]["environment_actions"]
    require(chunk["environment_actions"] == executed, "Selected/executed chunk differs")
    require([row["action"] for row in block] == executed[:len(block)], "Trace/action binding failed")
    return {"doc": doc, "block": block, "chunk": chunk, "candidates": candidates,
            "selected": selected, "success": ever, "steps": count,
            "capped_steps": count if ever else 300,
            "sources": {"episode": relative(trace_path.with_suffix(".json")),
                        "trace": relative(trace_path), "first_query": relative(chunk_path)}}


def analyze():
    swap, utility, audit, scene, config = map(read, (SWAP, UTILITY, PRIOR_AUDIT, SCENE, CONFIG))
    require(swap["validated_pairs"] == 30 and not swap["validation_errors"], "Swap panel unqualified")
    require(utility["complete_amended_grid"] and utility["validated_branches"] == 60
            and utility["selected_arm_full_source_parity_count"] == 30, "Utility panel unqualified")
    require(audit["status"] == "passed" and audit["comparison"]["all_metrics_match"]
            and not audit["comparison"]["differences"], "Prior independent audit not passed")
    require(audit["checks"]["original Goal native N2 300 budget"] == 60
            and audit["checks"]["subsequent selection is native argmax"] == 398,
            "Missing previously audited original-Goal continuation")
    expected = {(task, init) for task in range(10) for init in range(3)}
    swap_rows = {(r["task_id"], r["init_id"]): r for r in swap["rows"]}
    branches = {(r["task_id"], r["init_id"], r["forced_candidate_index"]): r
                for r in utility["episode_audits"]}
    registrations = {(r["task_id"], r["init_id"]): r for r in config["planned_episodes"]}
    require(len(swap["rows"]) == len(swap_rows) == 30 and set(swap_rows) == expected,
            "Incomplete/duplicate Swap grid")
    require(len(utility["episode_audits"]) == len(branches) == 60
            and set(branches) == {(t, i, c) for t, i in expected for c in (0, 1)},
            "Incomplete/duplicate measured branch grid")
    require(set(registrations) == expected, "Registration grid differs")
    scene_pairs = {r["original_task_id"]: r for r in scene["cyclic_pairs"]}
    rows = []
    for task, init in sorted(expected):
        source = swap_rows[task, init]
        registered = registrations[task, init]
        artifacts = source["audit"]["paired_artifact_hashes"]
        episodes = {arm: episode(artifacts[arm]["trace"]["path"]) for arm in ("clean", "swap")}
        for index in (0, 1):
            ref = branches[task, init, index]
            measured = episode(ref["artifacts"]["trace"]["path"])
            require(measured["selected"] == index, "Forced candidate label differs from actual selection")
            require(measured["success"] == ref["success"] and measured["steps"] == ref["executed_steps"],
                    "Measured utility differs from prior audited branch")
            episodes[f"forced{index}"] = measured
        clean, swapped = episodes["clean"], episodes["swap"]
        original_goal = clean["doc"]["goal"]
        expected_candidates = [(c["index"], c["seed"], c["environment_actions"])
                               for c in clean["candidates"]]
        for arm, e in episodes.items():
            d = e["doc"]
            require((d["task_id"], d["init_id"]) == (task, init), "Episode root differs")
            require(d["goal"] == original_goal and d["seed"] == 31 and d["environment_seed"] == 0
                    and d["max_steps"] == 300 and d["settle_steps"] == 10 and d["best_of_n"] == 2,
                    "Goal or runtime protocol differs")
            require(d["initial_state_sha256"] == clean["doc"]["initial_state_sha256"],
                    "Recorded initial identity differs (existing identities, not rehashed)")
            require([(c["index"], c["seed"], c["environment_actions"]) for c in e["candidates"]]
                    == expected_candidates, "Index/seed/full-controls mapping differs")
            require(d["condition"] == ("goal_swap" if arm == "swap" else "clean"), "Wrong source condition")
            if arm != "swap":
                require(e["candidates"] == clean["candidates"], "Original-Goal candidate/value record differs")
        require(swapped["doc"]["policy_goal"] == swapped["doc"]["value_goal"] == original_goal,
                "Policy or Value Goal changed")
        require(registered["policy_goal"] == registered["value_goal"] == original_goal
                and swapped["doc"]["forecast_goal"] == registered["forecast_goal"], "Unregistered Goal routing")
        scene_pair = scene_pairs[task]
        require(scene_pair["same_parsed_physical_definition"] and scene_pair["different_goal_predicate"]
                and scene_pair["forecast_goal"] == registered["forecast_goal"], "Goal scene binding differs")
        matched = source["audit"]["matched_queries_only_before_state_divergence"][0]
        require(matched["first_step"] == 1 and matched["candidate_actions_equal"]
                and matched["clean_selected"] == clean["selected"]
                and matched["swap_selected"] == swapped["selected"], "Prior first-query audit differs")
        native_branch = episodes[f"forced{clean['selected']}"]
        derived_branch = episodes[f"forced{swapped['selected']}"]
        require(clean["block"] == native_branch["block"], "Clean/selected replay first physical block differs")
        require(swapped["block"] == derived_branch["block"], "Swap/selected replay first physical block differs")
        require((clean["success"], clean["steps"]) == (native_branch["success"], native_branch["steps"]),
                "Clean and audited native-selected continuation disagree")
        row = {"task_id": task, "init_id": init, "original_goal": original_goal,
               "forecast_goal_swap": registered["forecast_goal"],
               "clean_selected_index": clean["selected"], "swap_selected_index": swapped["selected"],
               "first_selection_flip": clean["selected"] != swapped["selected"],
               "clean_success": clean["success"], "clean_steps": clean["steps"],
               "clean_capped_steps": clean["capped_steps"],
               "continuous_swap_success": swapped["success"], "continuous_swap_steps": swapped["steps"],
               "derived_one_shot_success": derived_branch["success"],
               "derived_one_shot_steps": derived_branch["steps"],
               "derived_one_shot_capped_steps": derived_branch["capped_steps"],
               "derived_success_difference": int(derived_branch["success"]) - int(clean["success"]),
               "derived_capped_steps_difference": derived_branch["capped_steps"] - clean["capped_steps"],
               "candidate_identity_and_full_controls_exact": True,
               "clean_to_selected_replay_first_block_exact": True,
               "swap_to_selected_replay_first_block_exact": True,
               "same_declared_scene_distinct_legal_goal": True,
               "policy_and_value_goal_unchanged": True,
               "clean_goal_fields_binding": "Clean JSON stores goal; shared Policy/Future/Value routing and subsequent original-Goal argmax are inherited from the completed independent 60-branch audit, not newly recorded separate fields.",
               "sources": {arm: e["sources"] for arm, e in episodes.items()}}
        for index in (0, 1):
            e = episodes[f"forced{index}"]
            row.update({f"candidate{index}_success": e["success"], f"candidate{index}_steps": e["steps"],
                        f"candidate{index}_capped_steps": e["capped_steps"],
                        f"candidate{index}_execution_version": branches[task, init, index]["execution_version"]})
        rows.append(row)
    lost = [r for r in rows if r["derived_success_difference"] < 0]
    metrics = {"fixed_roots": 30, "tasks": 10, "measured_continuation_branches_reused": 60,
               "first_selection_flip_count": sum(r["first_selection_flip"] for r in rows),
               "clean_success_count": sum(r["clean_success"] for r in rows),
               "derived_one_shot_success_count": sum(r["derived_one_shot_success"] for r in rows),
               "continuous_swap_success_count_separate": sum(r["continuous_swap_success"] for r in rows),
               "derived_lost_clean_success_count": len(lost),
               "derived_gained_clean_failure_count": sum(r["derived_success_difference"] > 0 for r in rows),
               "derived_total_capped_steps_difference": sum(r["derived_capped_steps_difference"] for r in rows),
               "derived_mean_capped_steps_difference": sum(r["derived_capped_steps_difference"] for r in rows) / 30}
    return {"schema_version": 1, "status": "completed_existing_cache_first_selection_join",
            "scope": "Post hoc development evidence joining already measured first selections to already measured original-Goal continuations. No new one-shot intervention episode was executed.",
            "sources": {"goal_swap_analysis": relative(SWAP), "candidate_utility_analysis": relative(UTILITY),
                        "completed_independent_branch_audit": relative(PRIOR_AUDIT),
                        "scene_audit": relative(SCENE), "original_intervention_registration": relative(CONFIG)},
            "source_reuse": "Reuse the completed 60-branch independent audit for full continuation, source parity and protocol qualification; newly bind the four first-query candidate/control records, first physical blocks and trace-derived stopping outcomes. No hashes recomputed.",
            "rng_and_routing_source_basis": {
                "future_only_routing": "scripts/pilot_cosmos_native_goal_intervention.py:52-63: action uses policy_text, Future uses forecast_text, Value uses policy_text.",
                "fixed_candidate_noise": "scripts/cosmos_policy_native_inference.py:171-184 and vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/noise_utils.py:13-25: each call constructs noise using a seed-local NumPy RandomState, with candidate seeds31/32.",
                "one_selection_override": "scripts/pilot_cosmos_first_candidate_utility.py:247-304: full original-Goal first query is computed, only selected_index is forced once, later queries use unchanged original-Goal best_of_n.",
                "limit": "No full-latent or RNG-state snapshot equality is newly established. Binding is to saved candidate index/seed/full executed controls, physical first block and the audited continuation implementation."},
            "metrics": metrics, "rows": rows,
            "harm_case_evidence": [{"task_id": r["task_id"], "init_id": r["init_id"],
                                    "clean_first_index": r["clean_selected_index"],
                                    "swap_first_index": r["swap_selected_index"],
                                    "clean_success_steps": [r["clean_success"], r["clean_steps"]],
                                    "original_goal_continuation_after_swap_selected_chunk_success_steps":
                                        [r["derived_one_shot_success"], r["derived_one_shot_steps"]],
                                    "first_chunk_physical_binding_exact": True,
                                    "sources": r["sources"]} for r in lost],
            "interpretation": [
                "The observed first 16-step candidate choice is sufficient for the recorded failure under the measured original-Goal continuation at the identified development root.",
                "This is an additional causal linkage of existing outcomes, not a new harm/recovery experiment or an independent confirmation.",
                "The native selector need not beat uniform selection on the complete panel for this bounded first-selection effect to exist.",
                "The original continuous-Swap extension gate remains failed and its No-Go is unchanged."],
            "limits": [
                "Derived one-shot utility is selected from real branch replays; no actual episode with a single swapped Future query was newly run.",
                "One harmful development root does not establish a population attack success rate, independent generalization or a new mechanism.",
                "Failure after subsequent original-Goal planning does not imply that later Swap calls have no effect or that first choice fully mediates the continuous-Swap result.",
                "No Bus-Cut, matched Control-Cut or decision-recovery defense is evaluated.",
                "Same-scene legal Goal replacement is an internal researcher-defined Future-context intervention; it does not establish external reachability, in-distribution conditional inputs or a universal Goal-independent forecasting contract.",
                "The torch compatibility adapter and incomplete full TransformerEngine parity limitations of the original native-style runs remain.",
                "The already audited 50 original plus10 recovery branches retain their execution-version labels; this join does not create additional independent episodes."],
            "activity": {"new_model_calls": 0, "new_environment_steps": 0, "new_api_requests": 0,
                         "new_trajectories": 0, "hashes_recomputed": 0},
            "reproduction": f"{sys.executable} scripts/analyze_cosmos_first_selection_mediation.py --verify-only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify-only", action="store_true", help="Recompute the join and print metrics without writing files.")
    args = parser.parse_args()
    csv_path = args.out.with_suffix(".csv")
    if not args.verify_only:
        require(not args.out.exists() and not csv_path.exists(), "Refusing to overwrite an existing artifact")
    result = analyze()
    if not args.verify_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for row in result["rows"]:
            flat = {k: v for k, v in row.items() if k not in ("sources", "clean_goal_fields_binding")}
            for arm, sources in row["sources"].items():
                flat[f"{arm}_trace"] = sources["trace"]
            csv_rows.append(flat)
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        result["csv"] = relative(csv_path)
        with args.out.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "files_written": [] if args.verify_only else [str(args.out), str(csv_path)],
                      "activity": result["activity"]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
