#!/usr/bin/env python3
"""Analyze the frozen one-root text-route diagnostic using saved evidence."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cosmos_action_text_route_gate_20261015"
REPORT = ROOT / "reports/cosmos_action_text_route_decision_20261015.json"
CSV = REPORT.with_suffix(".csv")
NAMES = ("clean", "swap", "clean_action_swap", "swap_action_clean",
         "clean_proprio_swap", "swap_proprio_clean")


def read(path):
    return json.loads(Path(path).read_text())


def analyze():
    summary = read(RUN / "summary.json")
    protocol = read(RUN / "launched_config.json")
    assert summary["status"] == "completed_calibrated_bidirectional_text_route"
    assert summary["calibration"]["all_passed"]
    assert all(summary["calibration"]["checks"].values())
    assert summary["activity"] == protocol["max_activity"]
    assert set(summary["conditions"]) == set(NAMES)
    utilities = []
    for i in range(2):
        path = (ROOT / "runs/cosmos_first_candidate_utility_matched30_v1/task03_init02"
                / f"candidate{i}/libero_goal_task03_init02_clean.json")
        episode = read(path)
        assert episode["task_id"] == 3 and episode["init_id"] == 2 and episode["seed"] == 31
        utilities.append({"candidate_index": i, "success": episode["success"],
                          "steps": episode["environment_steps"],
                          "capped_steps": episode["environment_steps"] if episode["success"] else 300,
                          "source": str(path.relative_to(ROOT))})
        assert {k: utilities[i][k] for k in summary["historical_continuation_utilities"][i]} == summary["historical_continuation_utilities"][i]
    rows = []
    for name in NAMES:
        candidates = summary["conditions"][name]
        assert [r["candidate_index"] for r in candidates] == [0, 1]
        scores = [r["value"] for r in candidates]
        selected = 0 if scores[0] >= scores[1] else 1
        background = "swap" if name.startswith("swap") else "clean"
        expected_background_choice = 0 if background == "swap" else 1
        for candidate in candidates:
            assert (RUN / candidate["future_file"]).is_file()
            if candidate["hook"]:
                hook = candidate["hook"]
                assert hook["rows_per_block"] == 196 and hook["nonselected_rows_exact"]
                assert len(hook["calls_by_block"]) == 28 and min(hook["calls_by_block"]) > 0
                assert len(set(hook["calls_by_block"])) == 1
        rows.append({"condition": name, "background": background, "scores": scores,
                     "candidate1_minus_candidate0_value": scores[1] - scores[0],
                     "selected_index": selected, "exact_tie": scores[0] == scores[1],
                     "flip_from_clean": selected != 1,
                     "flip_from_own_background": selected != expected_background_choice,
                     "selected_historical_utility": utilities[selected],
                     "visual_latent_mse": [r["visual_latent_mse"] for r in candidates],
                     "direct_replacement_rms": [r["hook"].get("replacement_rms") for r in candidates]})
    mapping = {r["condition"]: r for r in rows}
    assert mapping["clean"]["selected_index"] == 1 and mapping["swap"]["selected_index"] == 0
    strict = lambda name, wanted: not mapping[name]["exact_tie"] and mapping[name]["selected_index"] == wanted
    checks = {
        "action_injection_selects_bad_candidate": strict("clean_action_swap", 0),
        "action_repair_selects_good_candidate": strict("swap_action_clean", 1),
        "control_injection_preserves_clean_choice": strict("clean_proprio_swap", 1),
        "control_repair_preserves_swap_choice": strict("swap_proprio_clean", 0),
    }
    passed = all(checks.values())
    report = {
        "status": "completed_fixed_route_analysis", "question": protocol["question"],
        "protocol": str((RUN / "launched_config.json").relative_to(ROOT)),
        "source": str((RUN / "summary.json").relative_to(ROOT)),
        "runner": "scripts/probe_cosmos_action_text_route.py",
        "analysis_script": "scripts/analyze_cosmos_action_text_route.py",
        "design": "One post-hoc known harmful root; two fixed native candidates; six conditions; two identity-route calibrations. All 28 cross-attention blocks, fixed196-row action/proprio groups, same current query, fixed seed31/32 in both Future and Value. Only Future instrumented.",
        "table": rows, "historical_utilities": utilities,
        "frozen_selectivity_gate": checks, "action_position_selectivity_gate_passed": passed,
        "decision": ("GO only for the fixed local direct-text-entry selectivity signature; no panel expansion or core claim follows."
                     if passed else "NO-GO for the preregistered direct action-position selectivity explanation at this root; do not tune layers, token groups, strengths or seeds to recover it."),
        "activity": summary["activity"],
        "denoiser_forward_calls": summary["denoiser_forward_calls"],
        "extra_cross_attention_forwards": summary["extra_cross_attention_forwards"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_cuda_reserved_bytes": summary["peak_cuda_reserved_bytes"],
        "calibration_checks_passed": len(summary["calibration"]["checks"]),
        "limits": [
            "This tests direct text entry at action positions, not a complete action bus cut. Other positions may transmit Goal information to action positions through self-attention.",
            "Same token count is not equal information, activation magnitude or sensitivity. The fixed proprio control is one location control, not all alternatives.",
            "The intervention uses researcher access to internal activations and trusted original text. It is neither an established external attacker interface nor an independently deployable defense.",
            "Future latents and Value scores are newly measured; selected-action success and capped steps are derived from two previously executed original-goal continuations. No new full-task or recovery episodes.",
            "Two candidates share one post-selected physical root. Flip indicators are descriptive, not population rates; no significance test or confidence interval is justified.",
            "Forecast MSE is against native joint-VAE encodings of independently replayed physical consequences, not raw physics or task utility.",
            "All prior local Goal-Swap harm, truth restoration, failed30-root expansion and fixedK8 coupling negative results remain unchanged.",
        ],
        "core_security_claim": None, "research_goal": "active_unachieved",
        "csv": str(CSV.relative_to(ROOT)),
        "reproduce_analysis": "python scripts/analyze_cosmos_action_text_route.py --verify-only",
    }
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["condition", "value0", "value1", "margin1minus0", "selected", "tie",
                     "flip_from_clean", "historical_success", "historical_capped_steps", "mse0", "mse1", "rms0", "rms1"])
    for r in rows:
        writer.writerow([r["condition"], *r["scores"], r["candidate1_minus_candidate0_value"],
                         r["selected_index"], r["exact_tie"], r["flip_from_clean"],
                         r["selected_historical_utility"]["success"], r["selected_historical_utility"]["capped_steps"],
                         *r["visual_latent_mse"], *r["direct_replacement_rms"]])
    return report, output.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result, csv_text = analyze()
    if args.verify_only:
        assert read(REPORT) == result
        assert CSV.read_text() == csv_text.replace("\r\n", "\n")
        print("Saved analysis exactly reproduces from cached evidence.")
    else:
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        CSV.write_text(csv_text)
        print(json.dumps({"decision": result["decision"], "gate": result["frozen_selectivity_gate"],
                          "choices": {r["condition"]: r["selected_index"] for r in result["table"]}}, ensure_ascii=False))
