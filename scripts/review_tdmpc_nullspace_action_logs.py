#!/usr/bin/env python3
"""Independently review saved nullspace-action data and returns on CPU.

No checkpoint, agent, environment, optimizer, or CUDA operation is used.
Non-dynamics equality is a recorded runtime assertion, not a second weight audit.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/tdmpc_nullspace_action_logs_20261015"
DEFAULT_OUTPUT = ROOT / "reports/tdmpc_nullspace_action_logs_review_20261015.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def path_name(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def audit(run):
    summary = read(run / "summary.json")
    protocol = read(run / "frozen_protocol.json")
    report = {
        "schema_version": 1,
        "status": "incomplete_source_no_scientific_verdict",
        "source_status": summary["status"],
        "source_run": path_name(run),
        "activity": {"new_model_calls": 0, "new_environment_steps": 0,
                     "new_training_updates": 0, "model_api_calls": 0, "cuda_calls": 0},
        "conditional_harm_gate_passed": None,
        "method": "Independent CPU reads of saved data tensors and JSON; no checkpoint or environment loading.",
        "reproduction": ".venv-native-local/bin/python scripts/review_tdmpc_nullspace_action_logs.py --verify-only",
    }
    if summary["status"] != "completed":
        report["reason"] = "Partial or failed execution is not a negative task result; completed-arm values do not establish the four-arm gate."
        return report

    import torch

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def same(a, b, message):
        require(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), message)

    def tensor(t, shape, name):
        require(isinstance(t, torch.Tensor) and list(t.shape) == shape
                and t.device.type == "cpu" and bool(torch.isfinite(t).all()), name)

    arms = protocol["conditions"]
    require(arms == ["clean", "zero", "correlated", "permuted"]
            and set(summary["conditions"]) == set(arms), "Registered four arms differ")
    source_path = ROOT / protocol["source"]["path"]
    source = load(source_path)
    tensor(source["obs"], [301, 24], "Source observation schema")
    tensor(source["action"], [300, 6], "Source action schema")
    tensor(source["reward"], [300], "Source reward schema")
    require(int(source["task"]) == 4 and int(source["seed"]) == 21,
            "Source task/seed differs")
    require(bool((source["action"][:, 2:] == 0).all()), "Source tails are not zero")
    require(bool((source["action"].abs() <= 1).all()), "Source action bound")
    variants = load(run / "action_variants.pt")
    require(set(variants["actions"]) == {"zero", "correlated", "permuted"}, "Saved action variants differ")
    permutation = variants["permutation"]
    require(permutation.dtype == torch.int64 and list(permutation.shape) == [300]
            and sorted(permutation.tolist()) == list(range(300)), "Not a permutation")
    expected_perm = torch.randperm(300, generator=torch.Generator().manual_seed(17))
    same(permutation, expected_perm, "Seed17 complete-row permutation differs")
    require(protocol["source"]["permutation_seed"] == 17, "Permutation protocol differs")
    action_checks = {}
    for arm, action in variants["actions"].items():
        tensor(action, [300, 6], arm + " action schema")
        require(bool((action.abs() <= 1).all()), arm + " action bound")
        same(action[:, :2], source["action"][:, :2], arm + " changed executed coordinates")
        mask = action.new_tensor([1, 1, 0, 0, 0, 0])
        same(action * mask, source["action"], arm + " mask does not recover source")
        action_checks[arm] = {"active_actions_exact": True, "mask_recovers_source_exact": True,
                             "bounded_finite": True}
    zero, correlated, permuted = (variants["actions"][k] for k in arms[1:])
    same(zero, source["action"], "Zero arm differs from historical source")
    same(correlated[:, 2:], source["action"][:, :2].repeat(1, 2), "Correlated encoding differs")
    same(permuted[:, 2:], correlated[permutation, 2:], "Permuted joint rows differ")
    multiset = lambda a: collections.Counter(tuple(row) for row in a.tolist())
    require(multiset(correlated[:, 2:]) == multiset(permuted[:, 2:]), "Joint tail multiset differs")

    replays = {}
    for arm in ["zero", "correlated"]:
        replay = load(run / f"{arm}_physical_replay.pt")
        same(replay["obs"], source["obs"], arm + " source replay observations differ")
        same(replay["reward"], source["reward"], arm + " source replay rewards differ")
        require(replay["done"] == [False] * 300, arm + " unexpected source termination")
        require(len(replay["native_rewards"]) == 300, arm + " source reward count")
        same(torch.tensor(replay["native_rewards"], dtype=replay["reward"].dtype),
             replay["reward"], arm + " physical replay reward list differs")
        replays[arm] = replay
    require(replays["zero"]["native_rewards"] == replays["correlated"]["native_rewards"],
            "Paired native source reward lists differ")
    require(summary["physical_equivalence"]["passed"] is True, "Runtime physical-equivalence gate absent")
    starts = load(run / "training_starts.pt")
    require(starts.dtype == torch.int64 and list(starts.shape) == [40, 32]
            and int(starts.min()) >= 0 and int(starts.max()) <= 297, "Training index bank schema")

    eval_spec = protocol["evaluation"]
    clean_history_path = ROOT / "runs/tdmpc_adaptation_screen_v0/clean_reacher-easy.pt"
    zero_history_path = ROOT / eval_spec["historical_trace"]
    histories = {"clean": load(clean_history_path), "zero": load(zero_history_path)}
    learning_history_path = ROOT / eval_spec["historical_learning_log"]
    old_learning = read(learning_history_path)["updates"]
    require(len(old_learning) == 40, "Historical update count differs")
    returns, evaluation_rows, training_checks, baseline_checks = {}, [], {}, {}
    for arm in arms:
        record = summary["conditions"][arm]
        data = load(run / f"{arm}_evaluation.pt")
        physical = load(run / f"{arm}_physical_trace.pt")
        require(data["task"] == 4 and data["seed"] == 11, arm + " eval task/seed differs")
        for field, shape in [("obs", [501, 24]), ("action", [500, 6]), ("reward", [500])]:
            tensor(data[field], shape, arm + ": " + field)
        require(len(physical["obs"]) == 501 and len(physical["action"]) == len(physical["reward"]) == 500,
                arm + " physical trace is incomplete")
        same(torch.stack(physical["obs"]), data["obs"], arm + " physical observation trace differs")
        same(torch.stack(physical["action"]), data["action"][:, :2], arm + " physical executed actions differ")
        same(torch.tensor(physical["reward"], dtype=data["reward"].dtype), data["reward"],
             arm + " physical reward trace differs")
        require(bool((data["action"][:, 2:] == 0).all()) and bool((data["action"].abs() <= 1).all()),
                arm + " native evaluation action padding/bounds")
        same(data["obs"][0], histories["clean"]["obs"][0], arm + " paired root differs")
        physical_return = math.fsum(physical["reward"])
        reward_sum = float(data["reward"].double().sum())
        require(physical_return == reward_sum == record["evaluation"]["return"], arm + " real return differs")
        ev = record["evaluation"]
        require(ev["steps"] == 500 and ev["full_episode"] is True and ev["task"] == "reacher-easy"
                and ev["task_id"] == 4 and ev["seed"] == 11 and ev["policy"] == "mpc",
                arm + " complete native episode metadata differs")
        returns[arm] = reward_sum
        evaluation_rows.append({"arm": arm, "real_reward_sum_float64": reward_sum,
                                "steps": 500, "paired_initial_obs_exact": True,
                                "physical_trace_exact": True, "native_tail_zero": True})
        logs = record["learning"]
        require(len(logs) == (0 if arm == "clean" else 40), arm + " training dose differs")
        if arm != "clean":
            require([r["update"] for r in logs] == list(range(1, 41))
                    and all(math.isfinite(r[k]) for r in logs for k in ["weighted_consistency_loss", "grad_norm"]),
                    arm + " learning log invalid")
            require(record["non_dynamics_state_exact"] is True, arm + " non-dynamics attestation missing")
            training_checks[arm] = {"updates": 40, "non_dynamics_state_exact_recorded": True}
        if arm in histories:
            reference = histories[arm]
            require(reference["task"] == 4 and reference["seed"] == 11, arm + " historical metadata differs")
            same(reference["obs"][0], data["obs"][0], arm + " historical root differs")
            old_return = float(reference["reward"].double().sum())
            require(old_return == eval_spec[f"historical_{arm}_return"], arm + " historical return anchor differs")
            qualified = reward_sum >= eval_spec["minimum_clean_and_zero_return"] and abs(reward_sum-old_return) <= eval_spec["maximum_baseline_relative_return_change"]*abs(old_return)
            require(qualified and record["baseline_gate"]["passed"] is True, arm + " baseline gate fails")
            baseline_checks[arm] = {"passed": qualified, "historical_return": old_return,
                                   "actual_return": reward_sum,
                                   "full_historical_trace_equal_diagnostic": all(torch.equal(data[k], reference[k]) for k in ["obs", "action", "reward"])}
        if arm == "zero":
            fields = ["weighted_consistency_loss", "grad_norm"]
            errors = {k: max(abs(a[k]-b[k]) for a,b in zip(logs, old_learning)) for k in fields}
            historical_ok = all(math.isclose(a[k], b[k], rel_tol=eval_spec["history_rtol"], abs_tol=eval_spec["history_atol"]) for a,b in zip(logs, old_learning) for k in fields)
            require(historical_ok and record["historical_learning_reconstruction"]["passed"] is True,
                    "Zero historical learning fails registered tolerance")
            require(errors == record["historical_learning_reconstruction"]["maximum_absolute_errors"],
                    "Historical learning error summary differs")
            training_checks[arm].update({"historical_all40_passed": True,
                                        "historical_all40_exact": all(v == 0 for v in errors.values()),
                                        "maximum_absolute_errors": errors})

    losses = {k: 1-returns["correlated"]/returns[k] for k in ["zero", "permuted"]}
    gate = all(x["passed"] for x in baseline_checks.values()) and all(v >= .10 for v in losses.values())
    decision = summary["decision"]
    require(decision["returns"] == returns and decision["relative_loss_correlated"] == losses
            and decision["conditional_harm_gate_passed"] == gate
            and decision["permuted_minus_correlated_return"] == returns["permuted"]-returns["correlated"],
            "Reported decision differs from independent real-reward computation")
    expected_counts = {"checkpoint_loads": 1, "training_updates": 120, "source_replay_controls": 600,
                       "evaluation_controls": 2000, "explicit_environment_resets": 6,
                       "evaluation_episodes_completed": 4, "model_api_requests": 0}
    require(summary["activity"] == expected_counts, "Reported execution counts differ")
    report.update({
        "status": "completed_independent_cpu_cache_review",
        "data_controls": {"action_checks": action_checks, "joint_tail_multiset_exact": True,
                          "complete_row_permutation_seed17_exact": True, "encoding_uses_only_active_actions": True,
                          "physical_replay_observations_rewards_done_exact": True,
                          "source_task": 4, "source_seed": 21, "source_transitions": 300,
                          "protected_data_boundary": "Saved action variants preserve active actions/order. Both physical replays match source obs/reward exactly. All arms use one source obs/reward/task/seed object in launched runner; separate per-arm protected tensors were not saved. Source reward is not used by the consistency-only adaptation.",
                          "mask_is_input_identity_not_executed_defense": True},
        "evaluation_rows": evaluation_rows, "returns": returns,
        "baseline_checks": baseline_checks, "training_checks": training_checks,
        "minibatch_bank": {"shape": [40, 32], "valid_indices": True,
                           "boundary": "One saved CUDA-generated index bank reused by runner; no CUDA RNG regeneration in this CPU audit."},
        "relative_loss_correlated": losses,
        "permuted_minus_correlated_return": returns["permuted"]-returns["correlated"],
        "conditional_harm_gate_passed": gate,
        "decision": "conditional_dynamics_path_signal_requires_separate_qualification" if gate else "No-Go_fixed_correlation_specific_candidate",
        "registered_execution_counts": expected_counts,
        "frozen_state_evidence_boundary": "Verified non-dynamics equality records from launched runner. Full non-dynamics state snapshots were not saved; no independent tensor comparison or checkpoint rehash is claimed.",
        "statistics": {"independent_evaluation_roots": 1, "seed": 11, "p_value": None,
                       "confidence_interval": None, "scope": "Four paired closed-loop conditions, not four independent roots or fixed candidate-ranking comparisons."},
        "limitations": ["Added dynamics-only consistency adaptation, not native full joint continuation or a deployed data-ingestion service.",
                        "Forecast MSE is not recomputed and never substitutes for real utility.",
                        "Only the registered reacher task is evaluated; this is not cross-task victim harm.",
                        "A positive gate alone does not establish a new poisoning concept or deployable defense."],
        "source_paths": [path_name(x) for x in [run/"summary.json", run/"frozen_protocol.json", run/"launched_script.py",
                         run/"action_variants.pt", run/"training_starts.pt", source_path,
                         clean_history_path, zero_history_path, learning_history_path]],
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.run.resolve())
    if not args.verify_only:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result.get(k) for k in ["status", "returns", "relative_loss_correlated", "conditional_harm_gate_passed", "decision"]}, allow_nan=False))


if __name__ == "__main__":
    main()
