#!/usr/bin/env python3
"""Publish the fixed nullspace-log result from completed records; no inference."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/tdmpc_nullspace_action_logs_20261015"


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    protocol = json.loads((RUN / "frozen_protocol.json").read_text())
    if summary["status"] != "completed":
        raise RuntimeError("Do not publish an incomplete experiment as a completed result")
    rows = []
    for name in protocol["conditions"]:
        arm = summary["conditions"][name]
        result = arm["evaluation"]
        rows.append({"condition": name, "real_return": result["return"], "real_controls": result["steps"],
                     "one_step_latent_mse": arm["fixed_trace_one_step_latent_mse"],
                     "training_updates": len(arm["learning"]), "task": result["task"], "seed": result["seed"]})
    decision = summary["decision"]
    report = {
        "date": "2026-10-15",
        "status": "completed_fixed_real_utility_qualification",
        "question": protocol["question"],
        "protocol": "configs/tdmpc_nullspace_action_logs_20261015.json",
        "run": str(RUN.relative_to(ROOT)),
        "scope": protocol["scope"],
        "source_facts": {
            "executor": "MultitaskWrapper.step slices the action to the current task's physical dimensions.",
            "learner": "Buffer retains the full action; WorldModel.next concatenates all6 coordinates. Native pi/MPC mask generated tails, but the learner's action input is not masked.",
            "native_random_exception": "MultitaskWrapper.rand_act samples the full Box. The native multitask trainer is offline; this does not prove the released checkpoint saw nonzero tails or expose an online attacker.",
            "paths": ["vendor/tdmpc2/tdmpc2/envs/wrappers/multitask.py",
                      "vendor/tdmpc2/tdmpc2/common/buffer.py",
                      "vendor/tdmpc2/tdmpc2/common/world_model.py",
                      "vendor/tdmpc2/tdmpc2/train.py"]
        },
        "source_and_permission": protocol["source"],
        "data_control": summary["data_control"],
        "physical_equivalence": summary["physical_equivalence"],
        "result_table": rows,
        "decision": decision,
        "baseline_gates": {name: summary["conditions"][name]["baseline_gate"] for name in ["clean", "zero"]},
        "historical_learning_reconstruction": summary["conditions"]["zero"]["historical_learning_reconstruction"],
        "activity": summary["activity"],
        "execution": {"session_id": 30866, "exit_code": 0, "terminal_chunk_id": "a3e756",
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_cuda_reserved_bytes": summary["peak_cuda_reserved_bytes"],
                      "shared_gpu": True, "exclusive_gpu_requested": False},
        "interpretation": {
            "established": ["The source manipulation leaves the entire300-control physical trace unchanged, including actual observations and native rewards.",
                            "Clean and unedited continuation reproduce the competent historical real-task baseline.",
                            "The fixed conditional decision-harm gate is evaluated against BOTH an unedited and a joint-tail-multiset-matched control."],
            "not_established": ["A Goal-Swap or native full-training attack.",
                                "An externally writable deployed interface, population attack success rate, or generalization.",
                                "A new defense: task masking recovers the unedited training inputs exactly, but no separate defense episode was executed.",
                                "Novelty6.5–7 or an ICLR-ready core security mechanism."],
            "physical_vs_prediction": "The execution-equivalence contract concerns physical outcomes. Nonzero tails are outside this cache's original training representation; a learned conditional model need not already be invariant, and a latent error increase alone is not security harm.",
            "statistics": protocol["decision"]["statistics"],
            "closed_loop_boundary": "Candidates and later physical states may diverge across arms as part of the closed-loop effect. This experiment does not measure fixed-candidate ranking flips.",
            "existing_evidence_unchanged": "Preserve prior Cosmos local harm and failed expansion/route controls, DINO objective mismatch, and other bounded results. This new candidate does not erase or generalize them."
        },
        "prior_work": [
            {"title": "Causal confusion and spurious features in offline reinforcement learning",
             "url": "https://proceedings.mlr.press/v213/gupta23a.html",
             "use": "Conceptual precedent; executor-null coordinates alone do not establish a novel mechanism."},
            {"title": "Label-Consistent Backdoor Attacks",
             "url": "https://arxiv.org/abs/1912.02771",
             "use": "Input-feature poisoning can preserve correct labels; labels remaining truthful is not sufficient novelty."},
            {"title": "Shortcut Learning in Deep Neural Networks",
             "url": "https://arxiv.org/abs/2004.07780",
             "use": "Shortcut reliance and its failure to transfer are established concepts."}
        ],
        "historical_record_correction": {
            "location": "configs/world_model_padded_action_screen.json and runs/tdmpc_padded_action_shadowing_v0/summary.json",
            "issue": "Hard-coded historical walker MPC/random update returns972.297543/973.753003 disagree with the cited source cache's actual975.012229/974.915866. Both are benign relative to972.409103 Clean, so the old No-Go disposition is unchanged.",
            "new_gate": "This experiment uses same-task reacher Clean983/random-update984, verified directly from the corresponding cached traces and rebuilt before edits. Old walker numbers are not evidence for this gate."
        },
        "next_action": protocol["decision"]["next_if_positive"] if decision["conditional_harm_gate_passed"]
                       else protocol["decision"]["next_if_negative"],
        "scientific_goal_status": "active_unachieved_no_qualified_core_claim",
        "reproduction": [".venv-native-local/bin/python scripts/pilot_tdmpc_nullspace_action_logs.py",
                         "python scripts/analyze_tdmpc_nullspace_action_logs.py",
                         ".venv-native-local/bin/python scripts/review_tdmpc_nullspace_action_logs.py"],
        "reproduction_note": "Runner requires a new output directory to protect the archived run; copy the protocol with a new output path and bind the runner before any intentional rerun. The negative candidate does not justify rerunning automatically.",
        "review": "reports/tdmpc_nullspace_action_logs_review_20261015.json"
    }
    report["runner_maintenance_after_execution"] = {
        "preserved_execution_source": "runs/tdmpc_nullspace_action_logs_20261015/launched_script.py",
        "changes": ["Zero-return relative-loss denominator is now recorded as null and fails the dual-control harm gate.",
                    "An additional deadline check precedes the final completed marker."],
        "effect_on_saved_result": "Neither boundary was reached: both denominators984 and total140.823s below600s. No model or environment rerun."
    }
    destination = ROOT / "reports/tdmpc_nullspace_action_logs_decision_20261015.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with destination.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(destination.relative_to(ROOT)), "decision": decision,
                      "model_calls": 0, "environment_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
