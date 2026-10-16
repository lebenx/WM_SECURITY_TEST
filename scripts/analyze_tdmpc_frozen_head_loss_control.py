#!/usr/bin/env python3
"""Audit saved TD-MPC2 loss-control results using CPU tensors only.

No checkpoint, agent, environment, optimizer, or CUDA calls are made.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def audit(run):
    import torch

    summary = read(run / "summary.json")
    protocol = read(run / "launched_config.json")
    report = {
        "schema_version": 1,
        "status": "incomplete_source_no_scientific_verdict",
        "source_run": str(run.relative_to(ROOT)),
        "source_status": summary["status"],
        "method": "Independent CPU deserialization of evaluation/recording tensors and JSON logs; no model/environment loading or inference.",
        "activity": {"new_model_calls": 0, "new_environment_steps": 0,
                     "new_training_updates": 0, "model_api_calls": 0, "cuda_calls": 0},
        "rows": [],
        "statistics": {"independent_development_roots": 1, "evaluation_seed": 31,
                       "p_value": None, "confidence_interval": None,
                       "note": "Four arms share one previously used root; not four independent replications."},
    }
    if summary["status"] != "completed":
        report["reason"] = "Partial, deferred, invalid, or failed execution is not a task-failure result."
        return report
    expected = [c["id"] for c in protocol["conditions_in_order"]]
    require(set(summary["arms"]) == set(expected) and len(expected) == 4,
            "Exactly the four registered conditions must be present")
    initial = None
    historical_checks, frozen_checks, gradient_checks = {}, {}, {}
    for condition in protocol["conditions_in_order"]:
        arm_id = condition["id"]
        arm = summary["arms"][arm_id]
        folder = run / arm_id
        require(arm["status"] == "completed" and arm["source"] == condition["source"]
                and arm["loss"] == condition["loss"], arm_id + ": incomplete or wrong arm")
        data = torch.load(folder / "evaluation.pt", map_location="cpu", weights_only=True)
        recording = torch.load(folder / "recorded_trace.pt", map_location="cpu", weights_only=True)
        require(int(data["task"]) == int(recording["task"]) == 2
                and int(data["seed"]) == int(recording["seed"]) == 31,
                arm_id + ": task/seed mismatch")
        shapes = {"obs": [501, 24], "action": [500, 6], "reward": [500]}
        for field, shape in shapes.items():
            tensor = data[field]
            require(list(tensor.shape) == shape and tensor.device.type == "cpu"
                    and bool(torch.isfinite(tensor).all()), arm_id + ": invalid " + field)
            require(torch.equal(tensor, recording[field]), arm_id + ": recording mismatch " + field)
        require(bool((data["action"].abs() <= 1.000001).all()), arm_id + ": action bounds")
        if initial is None:
            initial = data["obs"][0].clone()
        require(torch.equal(initial, data["obs"][0]), arm_id + ": paired initial observation differs")
        reward_sum = float(data["reward"].to(torch.float64).sum())
        reward_fsum = math.fsum(float(v) for v in data["reward"].tolist())
        require(reward_sum == reward_fsum == arm["evaluation"]["return"],
                arm_id + ": saved real reward sum differs from reported return")
        require(arm["evaluation"]["steps"] == arm["physical_controls_recorded"] == 500
                and arm["evaluation"]["full_episode"] is True
                and arm["evaluation"]["task"] == "walker-run"
                and arm["evaluation"]["task_id"] == 2 and arm["evaluation"]["seed"] == 31
                and arm["evaluation"]["policy"] == "mpc", arm_id + ": full episode metadata")
        learning = read(folder / "learning.json")
        require(learning == arm["learning"] and len(learning) == 300,
                arm_id + ": learning log mismatch")
        require([r["update"] for r in learning] == list(range(1, 301)), arm_id + ": update sequence")
        require(all(math.isfinite(row[field]) for row in learning
                    for field in ("weighted_consistency_loss", "grad_norm", "total_loss")),
                arm_id + ": nonfinite training diagnostics")
        frozen_checks[arm_id] = {key: arm[key] for key in
                                ("frozen_before", "frozen_after_training", "frozen_after_evaluation")}
        require(all(v["exact"] is True and v["non_dynamics_state_entries_checked"] == 70
                    for v in frozen_checks[arm_id].values()), arm_id + ": frozen-state assertion missing")
        if condition["loss"] == "consistency":
            reference = read(ROOT / protocol["historical_reconstruction"]["learning_logs"][condition["source"]])
            require(len(reference) == 300, "Historical reference length")
            fields = protocol["historical_reconstruction"]["learning_all_300_steps"]["fields"]
            maxima = {field: max(abs(a[field] - b[field]) for a, b in zip(learning, reference)) for field in fields}
            require(all(a["update"] == b["update"] for a, b in zip(learning, reference)), "Historical indices")
            require(all(error == 0 for error in maxima.values()), arm_id + ": historical logs are not exact")
            old_return = protocol["historical_reconstruction"]["seed31_historical_returns"][condition["source"]]
            require(reward_sum == old_return, arm_id + ": historical return not exact")
            require(arm["learning_reconstruction"]["passed"] and arm["historical_return_check"]["passed"],
                    arm_id + ": runtime historical gate did not pass")
            historical_checks[arm_id] = {"updates": 300, "max_absolute_errors": maxima,
                                          "all_logged_fields_exact": True, "return_exact": True}
        else:
            check = arm["auxiliary_gradient_check"]
            require(math.isfinite(check["weighted_auxiliary_dynamics_grad_norm"])
                    and check["weighted_auxiliary_dynamics_grad_norm"] > 0
                    and check["reward_prediction_requires_grad"] and check["Q_prediction_requires_grad"]
                    and not check["target_requires_grad"] and check["separate_optimizer_steps"] == 0,
                    arm_id + ": auxiliary gradient evidence invalid")
            require(all(row["target_seed"] == 314159 + i and row["target_rng_restored"] is True
                        and math.isfinite(row["reward_loss"]) and math.isfinite(row["value_loss"])
                        for i, row in enumerate(learning)), arm_id + ": target RNG/loss evidence invalid")
            gradient_checks[arm_id] = {**check, "target_rng_restored_updates": 300}
        report["rows"].append({"arm": arm_id, "source": condition["source"], "loss": condition["loss"],
                               "seed": 31, "real_controls": 500, "real_reward_sum_float64": reward_sum,
                               "recorded_trace_exact": True, "paired_initial_observation_exact": True,
                               "observation_shape": [501, 24], "action_shape": [500, 6]})
    bank = torch.load(run / "batch_starts.pt", map_location="cpu", weights_only=True)
    require(list(bank["starts"].shape) == [300, 32] and int(bank["starts"].min()) >= 0
            and int(bank["starts"].max()) < 298 and bank["draw_calls"] == 300
            and bank["draw_shape"] == [32] and bank["seed"] == 31 and bank["high"] == 298,
            "Saved minibatch bank metadata invalid")
    returns = {row["arm"]: row["real_reward_sum_float64"] for row in report["rows"]}
    r0, c0, r1, c1 = (returns[k] for k in expected)
    g0, g1 = r0 - c0, r1 - c1
    conditions = {
        "old_gap_positive": {"actual": g0, "threshold": 0, "comparison": ">", "passed": g0 > 0},
        "new_gap_at_most_half_old_gap": {"actual": g1, "threshold": .5*g0, "comparison": "<=", "passed": g1 <= .5*g0},
        "cascade_gain_at_least_half_old_gap": {"actual": c1-c0, "threshold": .5*g0, "comparison": ">=", "passed": c1-c0 >= .5*g0},
        "random_preserves_at_least_95_percent": {"actual": r1, "threshold": .95*r0, "comparison": ">=", "passed": r1 >= .95*r0},
    }
    passed = all(c["passed"] for c in conditions.values())
    for key, value in {"G0": g0, "G1": g1, "interaction": g0-g1,
                       "random_source_change": r1-r0, "cascade_source_change": c1-c0,
                       "conditional_followup_signal": passed}.items():
        require(summary["analysis"][key] == value, "Reported analysis disagrees: " + key)
    require(summary["activity"]["optimizer_updates"] == 1200
            and summary["activity"]["physical_evaluation_controls"] == 2000,
            "Recorded execution totals differ")
    report.update({
        "status": "completed_independent_cpu_saved_data_review",
        "historical_reconstruction": historical_checks,
        "recorded_frozen_state_checks": frozen_checks,
        "recorded_auxiliary_gradient_checks": gradient_checks,
        "evidence_boundary": "Returns, shapes, paired initial observations, recording equality, historical log comparisons, and decision criteria were independently recomputed. Frozen-state equality and first-batch backward-gradient checks are verified runtime attestations from the reviewed runner; frozen non-dynamics tensors were not saved, so they are not independently rehashed or re-executed here. Historical weights were not saved either.",
        "batch_bank": {"shape": [300, 32], "valid_indices": True, "seed": 31,
                       "scope": "One saved bank reused by reviewed runner; no CUDA RNG regeneration in this audit."},
        "metrics": {"G0": g0, "G1": g1, "interaction": g0-g1,
                    "gap_reduction_fraction": (g0-g1)/g0,
                    "random_return_change": r1-r0, "random_relative_change": (r1-r0)/r0,
                    "cascade_return_change": c1-c0, "cascade_relative_change": (c1-c0)/c0},
        "registered_conditions": conditions,
        "selective_recovery_gate_passed": passed,
        "decision": "NO_GO_selective_recovery_claim" if not passed else "bounded_single_root_selective_recovery_observed",
        "interpretation": "The conditional gap is sensitive to the added native-form reward/value gradients. Gap reduction alone is not selective recovery: source-specific changes and all four frozen criteria determine the result. This is a protocol qualification control, not a new attack, universal defense, proof of non-normality, or independent replication. Closed-loop candidates may differ downstream; no same-state ranking-flip inference.",
        "artifact_paths": {"source_summary": str((run / "summary.json").relative_to(ROOT)),
                           "launched_protocol": str((run / "launched_config.json").relative_to(ROOT)),
                           "launched_runner": str((run / "launched_script.py").relative_to(ROOT))},
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/tdmpc_frozen_head_loss_control_20261015")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/tdmpc_frozen_head_loss_control_review_20261015.json")
    parser.add_argument("--verify-only", action="store_true", help="Audit and print a summary without writing any files")
    args = parser.parse_args()
    run, report_path = args.run.resolve(), args.report.resolve()
    destinations = [run / "analysis.json", run / "analysis_results.csv", report_path]
    if not args.verify_only:
        require(not any(p.exists() for p in destinations), "Refusing to overwrite an existing audit artifact")
    report = audit(run)
    report["reproduction"] = ".venv-native-local/bin/python scripts/analyze_tdmpc_frozen_head_loss_control.py --verify-only"
    if args.verify_only:
        print(json.dumps({"status": report["status"], "decision": report.get("decision"),
                          "metrics": report.get("metrics"), "verify_only": True, "files_written": 0}))
        return
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    for destination in (destinations[0], report_path):
        destination.write_text(text)
    with destinations[1].open("w", newline="") as stream:
        columns = ["arm", "source", "loss", "seed", "real_controls", "real_reward_sum_float64",
                   "recorded_trace_exact", "paired_initial_observation_exact"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps({"status": report["status"], "decision": report.get("decision"),
                      "metrics": report.get("metrics"), "report": str(report_path)}))


if __name__ == "__main__":
    main()
