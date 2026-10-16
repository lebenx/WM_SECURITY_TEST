"""Summarize the frozen oracle-coverage pilot without new model/environment calls."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/worldevolver_truthful_coverage_vendor_runtime_v2_20261016"
STEM = ROOT / "reports/worldevolver_truthful_coverage_decision_20261016"


def read(path: Path):
    return json.loads(path.read_text())


def main():
    summary = read(RUN / "actor_execution/summary.json")
    prepared = read(RUN / "prepare_summary.json")
    protocol = read(ROOT / "configs/worldevolver_truthful_coverage_vendor_runtime_v2_20261016.json")
    original_prepare = read(ROOT / "runs/worldevolver_truthful_coverage_20261016/prepare_summary.json")
    original_diagnostic = read(ROOT / "runs/worldevolver_truthful_coverage_replay_diagnostic_20261016/summary.json")
    fixed_diagnostic = read(ROOT / "runs/worldevolver_truthful_coverage_vendor_runtime_diagnostic_20261016/summary.json")
    earlier_controls = (original_prepare["environment_controls"]["explicit_including_initial_look"]
                        + original_diagnostic["environment_steps"]
                        + fixed_diagnostic["explicit_environment_controls"])
    preparation_controls = prepared["environment_controls"]["explicit_including_initial_look"]
    calls = [json.loads(line) for line in (RUN / "actor_execution/calls.jsonl").read_text().splitlines()]
    assert summary["status"] == "completed" and prepared["qualified"]
    assert len(calls) == summary["model_api_attempts"] == protocol["api_attempt_cap"] == 8
    assert all(c["http_status"] == 200 for c in calls)
    assert all(c["response"]["choices"][0]["finish_reason"] == "stop" for c in calls)
    by_tag = {c["tag"]: c for c in calls}
    assert len(by_tag) == len(calls)
    rows, signal_roots, flipped_roots = [], [], []
    for root in summary["roots"]:
        variation = root["variation"]
        branches = {b["action"]: b for b in read(RUN / f"variation_{variation}/branches.json")}
        prefix = f"variation_{variation}/"
        assert by_tag[prefix + "draft"]["request"] == by_tag[prefix + "sham"]["request"]
        assert read(RUN / f"variation_{variation}/reproduction.json")["full_object_tree_and_context_equal"]
        conditions = {"draft": root["draft"], **root["arms"]}
        for name, outcome in conditions.items():
            branch = branches[outcome["action"]]
            assert outcome["utility_score"] == branch["utility_score"]
            assert outcome["success"] == branch["success"]
            rows.append({"variation": variation, "condition": name, "selected_action": outcome["action"],
                         "q_fixed_continuation": branch["utility_score"], "success": branch["success"],
                         "action_changed_from_draft": outcome["action"] != root["draft"]["action"],
                         "utility_difference_from_draft": branch["utility_score"] - root["draft"]["utility_score"]})
        q = {k: v["utility_score"] for k, v in conditions.items()}
        signal = (q["draft_only_truth"] < min(q["draft"], q["sham"])
                  and q["all_candidate_truth"] >= max(q["draft"], q["sham"]))
        assert signal == root["candidate_signal"]
        if signal:
            signal_roots.append(variation)
        if conditions["draft_only_truth"]["action"] != conditions["draft"]["action"]:
            flipped_roots.append(variation)
    report = {
        "schema_version": 1, "date": "2026-10-16",
        "question": "Can truthful foresight covering only the draft action lower real selected-action utility, with all-candidate foresight restoring both no-foresight baselines?",
        "status": "completed", "decision": "fixed_pilot_no_signal_stop" if not signal_roots else "initial_signal_requires_independent_followup",
        "protocol": str(Path("configs") / "worldevolver_truthful_coverage_vendor_runtime_v2_20261016.json"),
        "runner": "scripts/probe_worldevolver_truthful_coverage.py",
        "analysis": "scripts/analyze_worldevolver_truthful_coverage.py",
        "independent_review": "reports/worldevolver_truthful_coverage_result_review_20261016.json",
        "run": str(RUN.relative_to(ROOT)),
        "scope": summary["scope"],
        "native_and_added": "Native WorldEvolver WMReactAgent prompt/parser/Foresight methods; researcher-added three-candidate restriction, single-generation cap and oracle forecasts. No full native WorldEvolver episode or pretrained world-model prediction evaluated.",
        "model_requested": protocol["model"],
        "model_response_labels": sorted({c["response"]["model"] for c in calls}),
        "activity": {"model_api_attempts": len(calls), "http_200": len(calls), "actor_elapsed_seconds": summary["elapsed_seconds"],
                     "preparation_explicit_environment_controls": preparation_controls,
                     "failed_preparation_and_runtime_diagnostic_explicit_controls": earlier_controls,
                     "all_stages_explicit_environment_controls": earlier_controls + preparation_controls,
                     "native_gold_generation_returned_action_history_length": prepared["environment_controls"]["native_gold_generation_action_history_length"],
                     "gold_history_count_caveat": "Not an instrumented count of all internal simulator operations; kept separate from explicit controls.",
                     "environment_controls_during_api_phase": summary["new_environment_controls"], "gpu_jobs": 0, "training_updates": 0},
        "results": rows,
        "fixed_panel_counts": {"roots": len(summary["roots"]), "strict_signal_roots": len(signal_roots),
                               "draft_only_action_flip_roots": len(flipped_roots)},
        "checks": {"full_roots_reproduced": True, "draft_sham_requests_equal": True,
                   "all_selected_utilities_match_environment_branches": True, "all_responses_finished_stop": True},
        "runtime_correction": protocol["runtime_revision"],
        "limitations": protocol["limitations"] + [
            "Two pour candidates have identical immediate observation text. Oracle truth at this interface omits the later mixing consequence.",
            "Temperature zero does not establish service-level seed determinism. Sham controls an additional identical-prompt draw.",
            "Q_fixed_continuation is independent environment scoring, not full autonomous return, irreversible harm or a safety violation.",
            "Selected-action identity is measured; complete action rankings were not elicited.",
            "No p-value, population effect, attack success rate, defense effectiveness or novelty score follows from two same-family development roots."
        ],
        "scientific_conclusion": "All four conditions chose the same score-100 action on both fixed roots. The predefined harm-and-recovery signal is absent; this does not establish universal coverage safety or falsify all foresight-consumer failures.",
        "next_action": "Stop this fixed candidate without task/root/seed/forecast-horizon search. Reuse native actor and exact-environment replay assets for materially different, independently qualified questions.",
        "related_source_screen": "reports/new_carrier_source_qualification_20261016.json",
        "research_goal_status": "active_unachieved",
    }
    STEM.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with STEM.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"decision": report["decision"], "counts": report["fixed_panel_counts"], "rows": len(rows)}))


if __name__ == "__main__":
    main()
