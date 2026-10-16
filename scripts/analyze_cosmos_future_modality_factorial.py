#!/usr/bin/env python3
"""Stdlib-only analysis of the frozen nine-cell Cosmos future-slot diagnostic."""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cosmos_future_modality_factorial_20261016.json"
REPORT = ROOT / "reports/cosmos_future_modality_factorial_decision_20261016.json"
CSV = REPORT.with_suffix(".csv")
LEVELS = ("clean", "swap", "truth")
ERROR_KEYS = ("future_proprio_slot_mse", "visual_latent_mse",
              "future_wrist_slot_mse", "future_primary_slot_mse", "all_future_slots_mse")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def name(p, v):
    return f"p_{p}_v_{v}"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def utility_from_episode(path, candidate, original_goal):
    ep = read(ROOT / path)
    expected = {"status": "completed", "task_id": 3, "init_id": 2, "goal": original_goal,
                "condition": "clean", "max_steps": 300, "best_of_n": 2,
                "seed": 31, "environment_seed": 0}
    require(all(ep.get(k) == v for k, v in expected.items()), f"Historical episode protocol differs: {path}")
    require(type(ep["success"]) is bool and type(ep["environment_steps"]) is int,
            "Historical outcome has invalid type")
    return {"candidate_index": candidate, "success": ep["success"],
            "steps": ep["environment_steps"],
            "capped_steps": ep["environment_steps"] if ep["success"] else 300,
            "episode_path": path}


def contrast(before, after):
    a, b = before["historical_selected_utility"], after["historical_selected_utility"]
    return {"from": before["name"], "to": after["name"],
            "candidate_value_differences": [after["values"][i] - before["values"][i] for i in range(2)],
            "margin_difference": after["candidate1_minus_candidate0_value"] - before["candidate1_minus_candidate0_value"],
            "selection_changed": before["selected_index"] != after["selected_index"],
            "both_orders_strict": not (before["exact_tie"] or after["exact_tie"]),
            "from_selected_index": before["selected_index"], "to_selected_index": after["selected_index"],
            "selected_historical_success_difference": int(b["success"]) - int(a["success"]),
            "selected_historical_capped_steps_difference": b["capped_steps"] - a["capped_steps"]}


def analyze():
    protocol = read(PROTOCOL)
    run = ROOT / protocol["output"]
    result = read(run / "summary.json")
    require(read(run / "launched_config.json") == protocol, "Launched protocol differs from current frozen protocol")
    require(result["status"] == "completed_calibrated_future_modality_factorial", "No completed factorial result; scientific comparisons unmeasured")
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2
            and protocol["candidate_indices"] == [0, 1] and protocol["candidate_seeds"] == [31, 32]
            and protocol["factor_levels"] == list(LEVELS), "Frozen root/candidates/factors differ")
    expected_conditions = [{"name": name(p, v), "P": p, "V": v} for p in LEVELS for v in LEVELS]
    require(protocol["conditions"] == expected_conditions, "Frozen nine-cell panel differs")
    require(set(result["conditions"]) == {c["name"] for c in expected_conditions}, "Missing or additional cells")
    require(set(result["ranking"]) == set(result["conditions"]), "Ranking panel differs")
    require(result["activity"] == protocol["max_activity"], "Completed activity does not equal frozen 18-call budget")
    require(result.get("active_call") is None, "Completed result has an active call")
    cal = result["calibration"]
    require(cal["all_passed"] is True and cal["diagonal_all_passed_before_mixed"] is True
            and cal["checks"] and all(v is True for v in cal["checks"].values()), "Diagonal calibration incomplete")
    require(result["checks"] and all(v is True for v in result["checks"].values()), "Protected input checks incomplete")
    technical_attempts = []
    resume = protocol.get("technical_resume")
    if resume:
        failed_dir = ROOT / resume["failed_output"]
        failed = read(failed_dir / "summary.json")
        original_protocol = read(ROOT / resume["original_frozen_protocol"])
        require({k: v for k, v in protocol.items() if k not in ("output", "technical_resume")}
                == {k: v for k, v in original_protocol.items() if k != "output"},
                "Technical resume changed scientific protocol fields")
        require(failed["status"] == "error" and failed["activity"] == resume["failed_activity"]
                and failed["activity"]["checkpoint_loads"] == 1
                and all(value == 0 for key, value in failed["activity"].items() if key != "checkpoint_loads")
                and failed["conditions"] == {} and not failed.get("checkpoint_paths")
                and failed["error"]["type"] == "ValueError"
                and failed["error"]["message"] == "f must be a string filename in order to use mmap argument"
                and resume["successful_checkpoint_loads_before_failure"] == 0,
                "Preserved startup failure is not the specified zero-inference technical attempt")
        technical_attempts.append({"summary": relative(failed_dir / "summary.json"), "status": failed["status"],
                                   "session_id": resume["failed_session"], "exit_code": resume["exit_code"],
                                   "elapsed_seconds": failed["elapsed_seconds"], "raw_activity": failed["activity"],
                                   "checkpoint_load_attempts": 1, "successful_checkpoint_loads": 0,
                                   "error": {k: failed["error"][k] for k in ("type", "message")},
                                   "interpretation": "Startup compatibility failure before any inference or scientific outcome; preserved, not treated as a negative result."})

    previous = read(ROOT / protocol["historical_utility_report"])
    source = read(ROOT / protocol["source_summary"])
    source_protocol = read(ROOT / previous["sources"]["protocol"])
    require(source["status"] == "completed_calibrated_first_query_truth_countercheck"
            and source["calibration"]["all_passed"] is True, "Source truth calibration incomplete")
    require(previous["status"] == "completed_cache_analysis", "Historical utility report incomplete")
    utilities = []
    for i, cached in enumerate(previous["historical_utilities"]):
        require(cached["candidate_index"] == i and i < 2, "Historical candidate binding differs")
        utility = utility_from_episode(cached["source_episode"], i, source_protocol["original_goal"])
        require({k: utility[k] for k in ("candidate_index", "success", "steps", "capped_steps")}
                == {k: cached[k] for k in ("candidate_index", "success", "steps", "capped_steps")},
                "Historical outcome report disagrees with actual episode")
        utilities.append(utility)
    require(len(utilities) == 2 and [(u["success"], u["steps"]) for u in utilities] == [(False, 300), (True, 170)],
            "Fixed historical branch outcomes differ")

    cells, table, call_indices = {}, [], []
    for condition in expected_conditions:
        label, p, v = condition["name"], condition["P"], condition["V"]
        records = result["conditions"][label]
        require(len(records) == 2, f"Candidate count differs in {label}")
        values = []
        for i, record in enumerate(records):
            require(record["candidate_index"] == i and record["seed"] == 31 + i
                    and record["P"] == p and record["V"] == v, f"Candidate/seed/factor binding differs: {label}")
            require(record["historical_utility"] == utilities[i], f"Historical utility join differs: {label}/{i}")
            require(finite_number(record["value"]), "Nonfinite Value")
            require(all(finite_number(record[k]) and record[k] >= 0 for k in ERROR_KEYS), "Invalid forecast error")
            require(math.isclose(record["visual_latent_mse"],
                                (record["future_wrist_slot_mse"] + record["future_primary_slot_mse"]) / 2,
                                rel_tol=1e-6, abs_tol=1e-12), "Visual MSE aggregation differs")
            require(p != "truth" or record["future_proprio_slot_mse"] == 0, "Truth P error is nonzero")
            require(v != "truth" or record["visual_latent_mse"] == 0, "Truth V error is nonzero")
            call_indices.append(record["value_call_index"])
            values.append(record["value"])
        exact_tie = values[0] == values[1]
        selected = 1 if values[1] > values[0] else 0
        margin = values[1] - values[0]
        rank = [selected, 1 - selected]
        cached_rank = result["ranking"][label]
        require(cached_rank["selected_index"] == selected and cached_rank["exact_tie"] == exact_tie
                and cached_rank["candidate1_minus_candidate0_value"] == margin
                and cached_rank["historical_selected_utility"] == utilities[selected],
                f"Cached native ranking/utility differs: {label}")
        cell = {**condition, "values": values, "candidate1_minus_candidate0_value": margin,
                "rank": rank, "selected_index": selected, "exact_tie": exact_tie,
                "selection_basis": "tie_driven_candidate0_priority" if exact_tie else "strict_value_order",
                "historical_selected_utility": utilities[selected],
                "candidate_forecast_errors": [{"candidate_index": i, **{k: r[k] for k in ERROR_KEYS}} for i, r in enumerate(records)]}
        cells[label] = cell
        for i, record in enumerate(records):
            table.append({"task_id": 3, "init_id": 2, "condition": label, "P": p, "V": v,
                          "candidate_index": i, "candidate_seed": 31 + i, "value": values[i],
                          "candidate1_minus_candidate0_value": margin, "exact_tie": exact_tie,
                          "rank_position": rank.index(i) + 1, "selected": selected == i,
                          "selected_index": selected, **{k: record[k] for k in ERROR_KEYS},
                          "historical_candidate_success": utilities[i]["success"],
                          "historical_candidate_steps": utilities[i]["steps"],
                          "historical_candidate_capped_steps": utilities[i]["capped_steps"],
                          "historical_selected_success": utilities[selected]["success"],
                          "historical_selected_capped_steps": utilities[selected]["capped_steps"],
                          "new_physical_execution": False})
    require(sorted(call_indices) == list(range(1, 19)), "Value call indices are not exactly 1..18")
    ordered_calls = [result["conditions"][label][i]["value_call_index"] for label in protocol["execution_order"] for i in range(2)]
    require(ordered_calls == list(range(1, 19)), "Diagonal-first execution order differs")
    for level, source_name in (("clean", "clean"), ("swap", "swap"), ("truth", "true_future")):
        require(cells[name(level, level)]["values"] == source["scores"][source_name], "Diagonal Value scores differ from calibrated source")
    for i in range(2):
        for p in LEVELS:
            require(len({result["conditions"][name(p, v)][i]["future_proprio_slot_mse"] for v in LEVELS}) == 1,
                    "P error changed when only V changed")
        for v in LEVELS:
            for key in ("visual_latent_mse", "future_wrist_slot_mse", "future_primary_slot_mse"):
                require(len({result["conditions"][name(p, v)][i][key] for p in LEVELS}) == 1,
                        "V error changed when only P changed")

    fixed_p = [{"fixed_P": p, **contrast(cells[name(p, a)], cells[name(p, b)])}
               for p in LEVELS for a, b in itertools.combinations(LEVELS, 2)]
    fixed_v = [{"fixed_V": v, **contrast(cells[name(a, v)], cells[name(b, v)])}
               for v in LEVELS for a, b in itertools.combinations(LEVELS, 2)]
    cc, cs, sc, ss = [cells[name(p, v)] for p, v in (("clean", "clean"), ("clean", "swap"), ("swap", "clean"), ("swap", "swap"))]
    def strict_select(cell, candidate):
        return not cell["exact_tie"] and cell["selected_index"] == candidate
    interaction = ss["candidate1_minus_candidate0_value"] - sc["candidate1_minus_candidate0_value"] - cs["candidate1_minus_candidate0_value"] + cc["candidate1_minus_candidate0_value"]
    truth_checks = {}
    for label, alternate, anchor in (("visual_swap_with_truth_proprio", name("truth", "swap"), name("truth", "clean")),
                                      ("proprio_swap_with_truth_visual", name("swap", "truth"), name("clean", "truth"))):
        cell, truth = cells[alternate], cells[name("truth", "truth")]
        truth_checks[label] = {"condition": alternate, "selected_index": cell["selected_index"],
                              "exact_tie": cell["exact_tie"],
                              "strictly_selects_historically_failed_candidate": strict_select(cell, 0),
                              "differs_strictly_from_all_truth_selection": not cell["exact_tie"] and not truth["exact_tie"] and cell["selected_index"] != truth["selected_index"],
                              "clean_to_swap_other_block_truth": contrast(cells[anchor], cell),
                              "all_truth_to_swap_other_block_truth": contrast(truth, cell)}
    findings = {"visual_bidirectional_clean_swap": strict_select(cs, 0) and strict_select(sc, 1),
                "proprio_bidirectional_clean_swap": strict_select(sc, 0) and strict_select(cs, 1),
                "crossed_cells_exact_tie": cs["exact_tie"] or sc["exact_tie"],
                "both_crossed_cells_strictly_select_failure": strict_select(cs, 0) and strict_select(sc, 0),
                "both_crossed_cells_strictly_select_success": strict_select(cs, 1) and strict_select(sc, 1),
                "exact_tie_cells": [label for label, cell in cells.items() if cell["exact_tie"]],
                "new_physical_episodes": 0,
                "decision": "Frozen local diagnostic complete; report every cell and close without searching new slots, cameras, strengths, seeds or roots."}
    if findings["visual_bidirectional_clean_swap"]:
        findings["crossed_cell_pattern"] = "The two crossed Clean/Swap cells follow the visual source in both strict ordering directions, conditional on these fixed input slots."
    elif findings["proprio_bidirectional_clean_swap"]:
        findings["crossed_cell_pattern"] = "The two crossed Clean/Swap cells follow the proprio source in both strict ordering directions, conditional on these fixed input slots."
    elif findings["crossed_cells_exact_tie"]:
        findings["crossed_cell_pattern"] = "At least one crossed cell is tied; neither registered strict bidirectional interpretation holds."
    else:
        findings["crossed_cell_pattern"] = "The crossed cells choose the same candidate; neither registered bidirectional modality pattern holds."
    report = {"schema_version": 1, "status": "completed_cache_analysis", "question": protocol["question"],
              "scope": protocol["scope"],
              "sources": {"protocol": relative(PROTOCOL), "launched_protocol": relative(run / "launched_config.json"),
                          "summary": relative(run / "summary.json"), "source_summary": protocol["source_summary"],
                          "historical_utility_report": protocol["historical_utility_report"],
                          "historical_episodes": [u["episode_path"] for u in utilities],
                          "analysis_script": "scripts/analyze_cosmos_future_modality_factorial.py"},
              "checks": {"frozen_protocol_exact": True, "nine_cells_two_candidates_complete": True,
                         "historical_utilities_match_episode_files": True, "calibration_checks": len(cal["checks"]),
                         "protected_input_checks": len(result["checks"]), "all_checks_passed": True,
                         "diagonal_calibration_before_mixed": True, "source_diagonal_values_exact": True,
                         "native_argmax_and_exact_ties_recomputed": True, "factor_error_consistency": True,
                         "call_count_and_execution_order_match_protocol": True},
              "historical_utilities": utilities, "cells": cells,
              "fixed_P_visual_contrasts": fixed_p, "fixed_V_proprio_contrasts": fixed_v,
              "clean_swap_interaction": {"formula": "m(S,S)-m(S,C)-m(C,S)+m(C,C), m=Value1-Value0",
                                         "margin_interaction": interaction,
                                         "candidate_value_interactions": [ss["values"][i] - sc["values"][i] - cs["values"][i] + cc["values"][i] for i in range(2)],
                                         "interpretation": "Numerical conditional dependence of fixed-seed Value outputs; not a mediation percentage or population estimate."},
              "truth_counterchecks": truth_checks, "findings": findings,
              "execution": {"elapsed_seconds": result.get("elapsed_seconds"), "activity": result["activity"],
                            "preserved_technical_attempts": technical_attempts,
                            "all_attempt_activity": {key: result["activity"][key] + sum(a["raw_activity"][key] for a in technical_attempts)
                                                     for key in result["activity"]},
                            "checkpoint_counter_semantics": "The raw checkpoint_loads counter counts attempts; completed run loaded two checkpoints successfully, while the preserved initial mmap failure loaded none successfully.",
                            "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
                            "peak_cuda_reserved_bytes": result.get("peak_cuda_reserved_bytes"),
                            "analysis": {"implementation": "Python standard library; saved JSON only",
                                         "model_calls": 0, "environment_steps": 0, "api_calls": 0, "weight_loads": 0}},
              "limits": protocol["limits"] + [
                  "Native Value is a discounted behavior-return proxy, not a calibrated success probability; fixed candidate seeds31/32 describe one realization.",
                  "Each selection is joined to a previously executed original-Goal continuation. No new closed-loop task or recovery episode occurred in this diagnostic.",
                  "Bidirectional flags follow the frozen strict ordering definitions. A tie is explicitly tie-driven and cannot establish either directional pattern.",
                  "Slot5 MSE is an error in the encoded proprio representation, not a parsed physical-state distance. All-slot MSE is descriptive only.",
                  "All contrasts reuse one post-hoc root and its two candidates; no independent statistical sample, ASR, confidence interval, universal invariance, or complete causal mediation is inferred."],
              "csv": relative(CSV), "reproduce": "python scripts/analyze_cosmos_future_modality_factorial.py --verify-only"}
    return report, table


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report, rows = analyze()
    table = csv_text(rows)
    if args.verify_only:
        require(read(REPORT) == report and CSV.read_text() == table, "Saved report/CSV differs from cache recomputation")
    else:
        require(not REPORT.exists() and not CSV.exists(), "Refusing to overwrite preserved analysis")
        REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        CSV.write_text(table)
    print(json.dumps({"status": "verified" if args.verify_only else "written", "report": relative(REPORT),
                      "selections": {k: v["selected_index"] for k, v in report["cells"].items()},
                      "findings": report["findings"]}, allow_nan=False))


if __name__ == "__main__":
    main()
