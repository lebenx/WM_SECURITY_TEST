#!/usr/bin/env python3
"""Recompute operator boundaries from saved traces; no native/model execution."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/safepred_branch_ownership_probe_20261015"
REPORT = ROOT / "reports/safepred_branch_ownership_decision_20261015.json"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    return json.loads(path.read_text())


def main():
    summary, trace = load(RUN / "summary.json"), load(RUN / "trace.json")
    fixtures = load(RUN / "fixtures_frozen_before_calls.json")
    protocol = load(RUN / "frozen_protocol.json")
    require(protocol == load(ROOT / "configs/safepred_branch_ownership_probe_20261015.json"),
            "Current protocol differs from executed frozen protocol")
    require(summary["status"] == "completed_cpu_native_operator_boundary_probe", "Incomplete probe")
    guidance = {row["case"]: row for row in trace["guidance"]}
    require(len(guidance) == 5 and len(trace["state_alias"]) == 6, "Unexpected case count")
    paired = []
    for prefix in ["multiroot_depth1", "single_root_depth2"]:
        left, right = [guidance[f"{prefix}_order{i}"] for i in range(2)]
        responses = []
        for row in [left, right]:
            events = [e for e in row["events"] if e["kind"] == "synthetic_response"]
            by_action = {e["action"]: e for e in events}
            require(len(by_action) == len(events), "Repeated action requires a richer event key")
            for action, event in by_action.items():
                require(event["delta"] == fixtures["deltas"][action], "Per-action response changed")
            responses.append(by_action)
        require(responses[0] == responses[1], "Per-action state, prompt or delta changed")
        require(left["selected_action"] == right["selected_action"] == "safe_root", "Selected action differs")
        for field in ["action", "risk", "path", "violated_policy_ids", "should_check_plan"]:
            require(left["native_selected_result"].get(field) == right["native_selected_result"].get(field),
                    "Selected-path field differs: " + field)
        require(left["selected_root_evaluation"] == right["selected_root_evaluation"], "Root metadata changed")
        for row in [left, right]:
            result = row["with_plan_result"]
            require(result["optimization_guidance"] == row["last_predicted_delta"]["optimization_guidance"],
                    "Guidance does not follow final delta")
            require(result["should_update_plan"], "Expected guidance update missing")
            require("path" not in row["wrapper_before_plan_result"]
                    and "should_check_plan" not in row["wrapper_before_plan_result"], "Wrapper boundary changed")
            require(row["plan_monitor_calls"] == 0, "Unexpected PlanMonitor call")
        require(left["with_plan_result"]["optimization_guidance"] != right["with_plan_result"]["optimization_guidance"],
                "No order dependence")
        paired.append({"pair": prefix, "per_action_inputs_prompts_deltas_exact": True,
                       "selected_action_and_path_exact": True, "guidance_differs": True,
                       "native_should_check_plan": left["native_selected_result"]["should_check_plan"]})
    require(not guidance["default_singleton_depth1"]["with_plan_result"]["should_update_plan"],
            "Singleton structural control differs")

    state_results, child_by_raw = [], {}
    for row in trace["state_alias"]:
        raw, arm = row["raw_input"], row["arm"]
        before, after = row["parent_before"], row["parent_after"]
        require("axtree_txt" in before and after["axtree_txt"] == before["axtree_txt"], "Tree changed or removed")
        require({k: v for k, v in before.items() if k != "key_elements"}
                == {k: v for k, v in after.items() if k != "key_elements"}, "Protected fields changed")
        require(row["child"] == child_by_raw.setdefault(raw, row["child"]), "Copy arm changed predicted child content")
        mutated = before["key_elements"] != after["key_elements"]
        require(mutated == (arm != "deepcopy_input"), "Isolation/control result differs")
        prompt_changed = {}
        for kind in ["world_model", "action"]:
            base = RUN / "prompts" / f"{raw}_{arm}_{kind}"
            prompt_changed[kind] = Path(str(base) + "_before.txt").read_bytes() != Path(str(base) + "_after.txt").read_bytes()
        require(not prompt_changed["action"], "Unexpected default action prompt dependence")
        require(prompt_changed["world_model"] == (mutated and not before["axtree_txt"]), "WM prompt boundary differs")
        state_results.append({"case": f"{raw}/{arm}", "parent_list_mutated": mutated,
                              "wm_prompt_changed": prompt_changed["world_model"], "action_prompt_changed": prompt_changed["action"]})

    replay_count = sum(len([e for e in r["events"] if e["kind"] == "synthetic_response"]) for r in guidance.values())
    require(replay_count == summary["activity"]["replay_simulate_calls"] == 15, "Response count mismatch")
    report = load(REPORT)
    report.update(status="completed_native_operator_probe_no_task_harm_measured",
                  execution={"exit_code": 0, "elapsed_seconds": summary["elapsed_seconds"],
                             "interpreter": summary["interpreter"], "attempts": 1},
                  activity=summary["activity"], guidance_results=summary["guidance_table"],
                  state_alias_results=state_results, trace_recomputed_pairs=paired,
                  statistics="Eleven deterministic code cases, not independent task trials; no ASR, effect-size population estimate or significance test.",
                  decision={"go": "Retain reproducible native routing and state-ownership boundary evidence.",
                            "no_go": "No decision-harm/security claim or paid task expansion follows from this operator result.",
                            "real_harm_signal": None, "dedicated_gpu_justified": False,
                            "reason": "No actual forecast, false factual attribution, low-trust writer, competent native Clean, environment loss or selective defense is established."},
                  code="scripts/probe_safepred_branch_ownership.py",
                  analysis_code="scripts/analyze_safepred_branch_ownership.py",
                  independent_review="reports/safepred_branch_ownership_review_20261015.json",
                  table="reports/safepred_branch_ownership_table_20261015.csv")
    report["harness_limits"] = summary["interpretation"]["limitations"] + [
        "The singleton control matches default search structure, not a full deployed Clean: synthetic policy IDs, empty policy list, fixed risk outputs and enabled long-term prompt construction bypass model response parsing and native rule-based risk scoring."]
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    columns = ["probe", "case", "parent_list_mutated", "wm_prompt_changed", "action_prompt_changed",
               "selected_action", "selected_path", "last_predicted_action", "should_update_plan"]
    with (ROOT / report["table"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in state_results:
            writer.writerow({"probe": "synthetic_state_operator", **row})
        for row in summary["guidance_table"]:
            writer.writerow({"probe": "synthetic_guidance_operator", **{k: row[k] for k in columns if k in row}})
    print(json.dumps({"status": report["status"], "paired_input_equality_checks": len(paired),
                      "state_cases": len(state_results), "synthetic_responses": replay_count,
                      "actual_task_utility": None, "model_api_calls": 0}))


if __name__ == "__main__":
    main()
