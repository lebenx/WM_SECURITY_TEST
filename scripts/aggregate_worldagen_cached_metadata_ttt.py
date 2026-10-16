#!/usr/bin/env python3
"""CPU-only aggregation of recorded WorldAgen cached-text TTT task outcomes.

Reads completed, versioned experiments; never imports a model or simulator.
The eight registered arms may arrive in stages. Missing arms are not failures
or successes. A recovery rate is defined only on actual NoLangTTT-success /
SwapTTT-failure pairs. Forecast losses and adapter equality are diagnostics,
never task utility. NoLangControlCutTTT was not registered or measured.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ("Frozen", "NoLangTTT", "GoalTTT", "SwapTTT", "EmptyTTT",
        "NoLangBusCutTTT", "SwapBusCutTTT", "SwapControlCutTTT")
SPLIT_VERSION = "six_action_purge1_disjoint_images_v2"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    # Exactly the identity serialization used by the experiment runner.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha(path)}


def validate_episode(ep, condition, protocol):
    """Verify actual trace-backed success and executed action count."""
    required = {"condition": condition, "suite": protocol["suite"], "task_id": protocol["task_id"],
                "seed": protocol["environment_seed"], "goal": protocol["clean_goal"],
                "model_device": protocol["model_device"]}
    for key, value in required.items():
        if type(ep.get(key)) is not type(value) or ep[key] != value:
            raise ValueError(f"Episode {condition} protocol mismatch: {key}")
    if (type(ep.get("success")) is not bool or ep.get("success_at_start") is not False
            or type(ep.get("terminal_done")) is not bool
            or type(ep.get("environment_steps")) is not int
            or not 1 <= ep["environment_steps"] <= protocol["max_steps"]):
        raise ValueError("Invalid/trivial episode or nonboolean task outcome")
    trace_path = Path(ep["trace"]).resolve()
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if len(trace) != ep["environment_steps"] or [r["step"] for r in trace] != list(range(1, len(trace) + 1)):
        raise ValueError("Episode trace action count/order differs")
    for row in trace:
        if (type(row.get("task_success")) is not bool or type(row.get("done")) is not bool
                or len(row["action"]) != 7 or not all(math.isfinite(x) for x in row["action"])):
            raise ValueError("Malformed physical action/outcome trace")
    if (trace[-1]["task_success"] != ep["success"] or trace[-1]["done"] != ep["terminal_done"]
            or any(r["task_success"] or r["done"] for r in trace[:-1])):
        raise ValueError("Summary success/termination does not match physical trace")
    return {"init_id": ep["init_id"], "success": ep["success"], "steps": ep["environment_steps"],
            "terminal_done": ep["terminal_done"], "source_trajectory_init": ep["init_id"] == protocol["source_init_id"],
            "initial_state_sha256": ep["initial_state_sha256"], "bddl_sha256": ep["bddl_sha256"],
            "trace": source_record(trace_path)}


def read_summary(path):
    path = Path(path).resolve()
    doc = load_json(path)
    if doc.get("status") not in ("completed", "stopped_no_lang_ttt_gate_failed"):
        raise ValueError(f"Refuse unfinished/resource-error summary {path}: {doc.get('status')}")
    p, data = doc["protocol"], doc["data"]
    identity = {k: v for k, v in p.items() if k != "description"}
    identity["cache_sha256"] = data["sha256"]
    if json_sha(identity) != doc.get("protocol_identity_sha256"):
        raise ValueError("Protocol/data identity hash differs")
    if (data.get("validation", {}).get("schema") != 3
            or p.get("data_split", {}).get("version") != SPLIT_VERSION):
        raise ValueError("Require schema3 purged physical cache")
    split = p["data_split"]
    if set(split["train_frame_indices"]) & set(split["heldout_frame_indices"]):
        raise ValueError("Training/heldout image indices overlap")
    if (p["train_windows"] != 30 or len(split["train_starts"]) != 30
            or len(split["heldout_starts"]) != p["heldout_windows"]
            or split["purged_starts"] != [180] or p["normal_adaptation_text"] != "no lang"):
        raise ValueError("Adaptation split or canonical-text protocol changed")
    opt = p["optimizer"]
    expected_opt = {"type": "AdamW", "lr": 5e-4, "weight_decay": .01,
                    "loss_scale": .1, "gradient_accumulation": 30, "updates": 1,
                    "clip_norm": .1, "clip_after_every_microbatch": True}
    if any(opt.get(k) != v for k, v in expected_opt.items()):
        raise ValueError("Registered single-update adaptation dose changed")
    if any(p["lora"].get(k) != v for k, v in {"rank": 64, "alpha": 32, "dropout": 0}.items()):
        raise ValueError("Registered LoRA dose changed")
    if doc.get("frozen_parameters_unchanged") is not True:
        raise ValueError("Frozen base parameter integrity was not verified")
    outputs = doc.get("zero_adapter_equivalence", {}).get("outputs", {})
    if set(outputs) != {"policy_arm", "policy_gripper", "world"} or not all(r.get("exact") is True for r in outputs.values()):
        raise ValueError("Zero-LoRA policy/world equivalence was not verified")
    cache_path = Path(data["path"])
    if file_sha(cache_path) != data["sha256"]:
        raise ValueError("Physical cache file changed")
    if any(a not in ("Frozen", "NoLangTTT") for a in doc["results"]):
        gate = doc.get("native_task_gate") or {}
        if (gate.get("passed") is not True or gate.get("episodes") != 30
                or gate.get("successes", 0) < 21 or gate.get("successful_tasks", 0) < 7):
            raise ValueError("Extended arm lacks the registered all-task native clean gate")
        gate_path = Path(gate["path"])
        if file_sha(gate_path) != gate["sha256"]:
            raise ValueError("Native clean-gate source changed")
        native = load_json(gate_path)
        expected = {"suite": p["suite"], "camera_size": p["camera_size"], "seed": p["environment_seed"],
                    "orientation": p["orientation"], "settle_steps": p["settle_steps"], "max_steps": p["max_steps"]}
        if native.get("status") != "completed" or any(native["protocol"].get(k) != v for k, v in expected.items()):
            raise ValueError("Native clean-gate physical protocol differs")
        native_rows = [r for r in native["results"] if r["condition"] == "native"]
        if (len(native_rows) != 30 or {(r["task_id"], r["init_id"]) for r in native_rows} != {(t, i) for t in range(10) for i in range(3)}
                or any(type(r.get("success")) is not bool or r.get("success_at_start") is not False for r in native_rows)
                or sum(r["success"] for r in native_rows) < 21
                or len({r["task_id"] for r in native_rows if r["success"]}) < 7):
            raise ValueError("Native clean gate does not meet the fixed all-task threshold")
    expected_literals = {"Frozen": "no lang", "NoLangTTT": "no lang", "GoalTTT": p["clean_goal"],
                         "SwapTTT": p["swap_goal"], "EmptyTTT": "", "NoLangBusCutTTT": "no lang",
                         "SwapBusCutTTT": p["swap_goal"], "SwapControlCutTTT": p["swap_goal"]}
    arms = {}
    for condition, result in doc["results"].items():
        if condition not in ARMS or result.get("condition") != condition:
            raise ValueError("Unknown or mislabeled adaptation arm")
        if result.get("adaptation_literal") != expected_literals[condition]:
            raise ValueError(f"Wrong cached text for {condition}")
        training = result["training"]
        if condition == "Frozen":
            if training["optimizer_updates"] != 0 or training["microbatches"] != 0:
                raise ValueError("Frozen condition was trained")
        elif (training["optimizer_updates"] != 1 or training["microbatches"] != 30
              or [r["start"] for r in training["rows"]] != split["train_starts"]):
            raise ValueError("Arm changed registered data order or adaptation dose")
        episodes = [validate_episode(ep, condition, p) for ep in result["episodes"]]
        ids = [ep["init_id"] for ep in episodes]
        if len(ids) != len(set(ids)) or set(ids) != set(p["eval_init_ids"]):
            raise ValueError("Completed arm must contain every registered init exactly once")
        success_count = sum(ep["success"] for ep in episodes)
        if result["task_successes"] != success_count or result["task_success_rate"] != success_count / len(episodes):
            raise ValueError("Arm success aggregate differs from physical traces")
        for name in ("heldout_before_adaptation_condition", "heldout_after_adaptation_condition",
                     "heldout_after_original_goal_native", "heldout_after_no_lang_native"):
            metric = result[name]
            if [r["start"] for r in metric["rows"]] != split["heldout_starts"]:
                raise ValueError("Forecast diagnostics use different heldout windows")
        arms[condition] = {"condition": condition, "episodes": episodes, "episode_count": len(episodes),
                           "successes": success_count, "success_rate": success_count / len(episodes),
                           "total_executed_steps": sum(e["steps"] for e in episodes),
                           "steps_successes_only": [e["steps"] for e in episodes if e["success"]],
                           "adaptation_literal": result["adaptation_literal"],
                           "adapter_tensor_sha256_recorded": result["adapter_sha256"],
                           "gradient_tensor_sha256_recorded": training.get("gradient_sha256"),
                           "adapter_file": source_record(path.parent / result["adapter_file"]),
                           "gradient_file": source_record(path.parent / result["gradient_file"]),
                           "forecast_error_diagnostics_only": {name: result[name]["image_mse_mean"] for name in
                               ("heldout_before_adaptation_condition", "heldout_after_adaptation_condition",
                                "heldout_after_original_goal_native", "heldout_after_no_lang_native")}}
    return {"summary": source_record(path), "identity": doc["protocol_identity_sha256"],
            "protocol": p, "arms": arms, "initial_adapter_sha256": doc["initial_adapter_sha256"],
            "normal_gate": doc["normal_adaptation_gate"], "native_gate": doc.get("native_task_gate"),
            "cache": source_record(cache_path),
            "recorded_frozen_parameter_sha256": doc["frozen_parameter_sha256_before"]}


def compare(arms, reference, intervention):
    if reference not in arms or intervention not in arms:
        return {"status": "not_measured", "missing_arms": [a for a in (reference, intervention) if a not in arms]}
    a = {r["init_id"]: r for r in arms[reference]["episodes"]}
    b = {r["init_id"]: r for r in arms[intervention]["episodes"]}
    if a.keys() != b.keys():
        raise ValueError("Paired arms have different initial states")
    rows = []
    for init in sorted(a):
        if any(a[init][k] != b[init][k] for k in ("initial_state_sha256", "bddl_sha256")):
            raise ValueError("Paired arms differ in physical initial state/task")
        both = a[init]["success"] and b[init]["success"]
        rows.append({"init_id": init, "reference_success": a[init]["success"],
                     "intervention_success": b[init]["success"], "reference_steps": a[init]["steps"],
                     "intervention_steps": b[init]["steps"],
                     "success_difference": int(b[init]["success"]) - int(a[init]["success"]),
                     "steps_difference_if_both_successful": b[init]["steps"] - a[init]["steps"] if both else None})
    costs = [r["steps_difference_if_both_successful"] for r in rows if r["steps_difference_if_both_successful"] is not None]
    return {"status": "measured_paired_environment_outcomes", "reference": reference, "intervention": intervention,
            "paired_initial_states": len(rows), "rows": rows,
            "lost_reference_success_init_ids": [r["init_id"] for r in rows if r["success_difference"] == -1],
            "gained_success_init_ids": [r["init_id"] for r in rows if r["success_difference"] == 1],
            "success_rate_difference_intervention_minus_reference": sum(r["success_difference"] for r in rows) / len(rows),
            "mean_steps_difference_both_success_only": statistics.mean(costs) if costs else None,
            "step_cost_pair_count": len(costs),
            "steps_scope": "Cost comparison only among paired cases succeeding in both arms; failed episode length is not a completion cost."}


def recovery(arms, defense):
    if any(a not in arms for a in ("NoLangTTT", "SwapTTT")):
        return {"status": "not_measured", "missing_arms": [a for a in ("NoLangTTT", "SwapTTT", defense) if a not in arms]}
    losses = compare(arms, "NoLangTTT", "SwapTTT")["lost_reference_success_init_ids"]
    if not losses:
        return {"status": "not_applicable_no_observed_attack_loss", "defense": defense,
                "defense_task_status": "measured" if defense in arms else "not_measured",
                "observed_attack_loss_init_ids": [], "recovered_init_ids": None,
                "recovery_fraction_of_observed_losses": None,
                "scope": "There is no observed loss denominator. This does not imply an unmeasured defense succeeded or was harmless."}
    if defense not in arms:
        return {"status": "not_measured", "defense": defense, "missing_arms": [defense],
                "observed_attack_loss_init_ids": losses, "recovery_fraction_of_observed_losses": None}
    successes = {r["init_id"]: r["success"] for r in arms[defense]["episodes"]}
    recovered = [i for i in losses if successes[i]]
    return {"status": "measured" if losses else "not_applicable_no_observed_attack_loss",
            "defense": defense, "observed_attack_loss_init_ids": losses,
            "recovered_init_ids": recovered, "recovery_fraction_of_observed_losses": len(recovered) / len(losses) if losses else None,
            "scope": "Recovery of actual NoLangTTT-success / SwapTTT-failure pairs only; no loss does not count as repair."}


def summarize_cohort(group, require_all=False):
    p, arms = group["protocol"], group["arms"]
    missing = [a for a in ARMS if a not in arms]
    if require_all and missing:
        raise ValueError(f"Missing registered arms: {missing}")
    comparisons = {name: compare(arms, a, b) for name, a, b in (
        ("normal_adaptation_vs_frozen", "Frozen", "NoLangTTT"),
        ("cached_text_poisoning_vs_no_lang", "NoLangTTT", "SwapTTT"),
        ("noncanonical_goal_metadata_diagnostic", "GoalTTT", "SwapTTT"),
        ("normal_bus_cut_cost", "NoLangTTT", "NoLangBusCutTTT"),
        ("residual_poisoning_under_bus_cut", "NoLangBusCutTTT", "SwapBusCutTTT"),
        ("bus_cut_vs_poisoned", "SwapTTT", "SwapBusCutTTT"),
        ("matched_control_cut_vs_poisoned", "SwapTTT", "SwapControlCutTTT"),
        ("bus_cut_vs_matched_control", "SwapControlCutTTT", "SwapBusCutTTT"),
        ("empty_canonicalization_normal_cost", "NoLangTTT", "EmptyTTT"),
        ("empty_canonicalization_vs_poisoned", "SwapTTT", "EmptyTTT"))}
    measured_gate = comparisons["normal_adaptation_vs_frozen"]
    gate_passed = (measured_gate["status"] != "not_measured"
                   and arms["Frozen"]["successes"] > 0
                   and not measured_gate["lost_reference_success_init_ids"])
    attack = comparisons["cached_text_poisoning_vs_no_lang"]
    attack_evidence = {"status": "not_measured"}
    if attack["status"] != "not_measured":
        lost = attack["lost_reference_success_init_ids"]
        attack_evidence = {
            "status": "observed_paired_success_loss" if lost else "no_observed_paired_success_loss",
            "observed_attack_loss_init_ids": lost,
            "success_loss_attack_supported_in_this_pilot": bool(lost),
            "interpretation": "Paired outcomes from one task and one adapted trajectory. A changed gradient, adapter or forecast alone does not establish task damage. Per-init step costs are secondary; this is not a general robustness claim.",
        }
    if any(a not in ("Frozen", "NoLangTTT") for a in arms) and not gate_passed:
        raise ValueError("Extended arms cannot be interpreted without matching measured normal health gate")
    tensor_diag = {"status": "not_measured"}
    if all(a in arms for a in ("NoLangBusCutTTT", "SwapBusCutTTT")):
        a, b = arms["NoLangBusCutTTT"], arms["SwapBusCutTTT"]
        tensor_diag = {"status": "recorded_digest_comparison", "adapter_tensor_sha256_equal": a["adapter_tensor_sha256_recorded"] == b["adapter_tensor_sha256_recorded"],
                       "gradient_tensor_sha256_equal": a["gradient_tensor_sha256_recorded"] == b["gradient_tensor_sha256_recorded"],
                       "scope": "Uses experiment-recorded tensor digests and separately records artifact file hashes; no unexecuted task success is inferred from equality."}
    return {"protocol_identity_sha256": group["identity"], "task_id": p["task_id"], "suite": p["suite"],
            "source_init_id": p["source_init_id"], "eval_init_ids": p["eval_init_ids"],
            "environment_seed": p["environment_seed"], "adapter_seed": p["adapter_seed"],
            "model_device": p["model_device"], "camera_size": p["camera_size"],
            "status": "all_eight_arms_observed" if not missing else "partial_observed_results",
            "observed_arms": [a for a in ARMS if a in arms], "missing_arms": missing,
            "normal_health_gate_recomputed": {"passed": gate_passed, "criterion": "At least one Frozen success and no Frozen success lost by NoLangTTT; not statistical equivalence."},
            "arms": arms, "paired_comparisons": comparisons,
            "task_success_damage_evidence": attack_evidence,
            "actual_loss_recovery": {a: recovery(arms, a) for a in ("SwapBusCutTTT", "SwapControlCutTTT", "EmptyTTT")},
            "bus_cut_tensor_diagnostic": tensor_diag,
            "missing_control_baseline": {"condition": "NoLangControlCutTTT", "status": "not_measured_not_registered", "normal_adaptation_cost": None,
                                         "limitation": "Control matches nine removed finite edges, but its normal-text adaptation/task cost was not measured. No clean-Control correction or Control difference-in-differences claim is supported."},
            "cluster_scope": {"tasks_in_this_cohort": 1, "adaptation_trajectories": 1,
                              "evaluation_initial_states": len(p["eval_init_ids"]),
                              "shared_adapted_parameters_across_eval_inits": True,
                              "source_init_is_development": True,
                              "heldout_forecast_windows_are_independent_episodes": False,
                              "ci_or_p_values_computed": False},
            "source_summaries": group["sources"], "physical_cache": group["cache"]}


def aggregate(paths, require_all=False, active_weight_audit=None):
    groups = {}
    for path in paths:
        item = read_summary(path)
        key = item["identity"]
        if key not in groups:
            groups[key] = {**item, "arms": {}, "sources": [], "frozen_digests": []}
        group = groups[key]
        if group["initial_adapter_sha256"] != item["initial_adapter_sha256"]:
            raise ValueError("Staged arms have different LoRA initialization")
        for arm, result in item["arms"].items():
            if arm in group["arms"]:
                raise ValueError(f"Duplicate arm {arm}; do not choose among repeated runs by outcome")
            group["arms"][arm] = result
        group["sources"].append(item["summary"])
        group["frozen_digests"].append({"summary": item["summary"]["path"],
                                        "recorded_tensor_sha256": item["recorded_frozen_parameter_sha256"]})
    audit_record = None
    if active_weight_audit:
        proof = load_json(active_weight_audit)
        if (proof.get("status") != "passed_cpu_cached_window_intervention"
                or proof.get("source_derived_active_weight_equivalence") is not True
                or proof.get("policy_world_loss_and_gradient_bitwise_equal") is not True):
            raise ValueError("Active-weight proof did not pass")
        if any(file_sha(path) != expected for path, expected in proof["source_files"].items()):
            raise ValueError("Active-weight proof sources changed")
        constructor_path = proof["constructor_audit_file"]
        if file_sha(constructor_path) != proof["constructor_audit_sha256"]:
            raise ValueError("Constructor proof changed")
        constructor = load_json(constructor_path)
        if (constructor["all_other_parameters_exact"] is not True
                or any(file_sha(path) != expected for path, expected in constructor["source_paths"].items())):
            raise ValueError("Constructor proof failed/source artifacts changed")
        if any(g["cache"]["sha256"] != proof["cache_file_sha256"] for g in groups.values()):
            raise ValueError("Active-weight proof uses a different physical cache")
        audit_record = source_record(active_weight_audit)
    cohorts = []
    for group in groups.values():
        cohort = summarize_cohort(group, require_all)
        whole_equal = len({x["recorded_tensor_sha256"] for x in group["frozen_digests"]}) == 1
        cohort["base_weight_pairing"] = {
            "recorded_full_frozen_parameter_digests": group["frozen_digests"],
            "recorded_full_frozen_parameter_digests_equal": whole_equal,
            "status": "recorded_full_digest_equal" if whole_equal else (
                "source_derived_active_weight_equivalence" if audit_record else "unresolved_full_digest_mismatch"),
            "active_weight_audit": audit_record,
            "direct_historical_active_parameter_digest_equality": False,
            "scope": "Whole-model digests are retained honestly. Source/CPU proof may exclude an unused randomly initialized MAE decoder; it does not reconstruct unsaved historical per-parameter digests.",
        }
        if not whole_equal and not audit_record:
            cohort["task_success_damage_evidence"]["causal_interpretation"] = "unresolved_base_weight_pairing; descriptive actual outcomes only"
        cohorts.append(cohort)
    return {"status": "observed_data_aggregation", "aggregator_sha256": file_sha(__file__),
            "threat_boundary": "Assumed write access to pretokenized cached .npy text; official CALVIN sampler canonicalizes to 'no lang'. This is reconstructed LIBERO TTT, not a native legal-Goal adaptation attack.",
            "statistical_scope": "Report within identical protocol/cache/device cohorts and paired task/init IDs. Arms and heldout windows are not independent samples. No cross-device pooling, CIs or population inference.",
            "no_forecast_substitution": "Task success/steps verified against real recorded environment traces. Forecast MSE, gradients and adapter hashes are secondary diagnostics only.",
            "cohorts": cohorts}


def self_test():
    def arm(values):
        return {"successes": sum(values), "episodes": [{"init_id": i, "success": value, "steps": 20 + i,
                 "initial_state_sha256": str(i), "bddl_sha256": "same"} for i, value in enumerate(values)]}
    arms = {"Frozen": arm([True, False, True]), "NoLangTTT": arm([True, False, True]),
            "SwapTTT": arm([False, True, True]), "SwapBusCutTTT": arm([True, True, True])}
    measured = compare(arms, "NoLangTTT", "SwapTTT")
    assert measured["lost_reference_success_init_ids"] == [0] and measured["gained_success_init_ids"] == [1]
    assert measured["success_rate_difference_intervention_minus_reference"] == 0
    assert measured["step_cost_pair_count"] == 1
    assert recovery(arms, "SwapBusCutTTT")["recovered_init_ids"] == [0]
    assert recovery(arms, "EmptyTTT")["status"] == "not_measured"
    arms["SwapTTT"] = arm([True, False, True])
    assert recovery(arms, "SwapBusCutTTT")["recovery_fraction_of_observed_losses"] is None
    assert recovery(arms, "SwapBusCutTTT")["status"] == "not_applicable_no_observed_attack_loss"
    assert recovery(arms, "EmptyTTT")["status"] == "not_applicable_no_observed_attack_loss"
    assert recovery(arms, "EmptyTTT")["defense_task_status"] == "not_measured"
    arms["SwapTTT"]["episodes"][0]["initial_state_sha256"] = "different"
    try:
        compare(arms, "NoLangTTT", "SwapTTT")
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched physical pair accepted")
    print(json.dumps({"status": "passed", "synthetic_cpu_only": True,
                      "checks": ["paired loss preserved despite equal aggregate success", "recovery only actual losses",
                                 "missing defense not measured", "no loss means no recovery denominator",
                                 "step cost only both-success pairs", "physical mismatch rejected"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-all-arms", action="store_true")
    parser.add_argument("--active-weight-audit", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.summaries or args.out_dir is None:
        parser.error("Require --summaries and --out-dir")
    result = aggregate(args.summaries, args.require_all_arms, args.active_weight_audit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty; preserve earlier reports")
    write_json(args.out_dir / "results.json", result)
    with (args.out_dir / "task_outcomes.csv").open("x", newline="") as stream:
        fields = ["protocol_identity", "suite", "task_id", "init_id", "condition", "model_device",
                  "success", "environment_steps", "source_trajectory_init"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort in result["cohorts"]:
            for condition, arm in cohort["arms"].items():
                for ep in arm["episodes"]:
                    writer.writerow({"protocol_identity": cohort["protocol_identity_sha256"], "suite": cohort["suite"],
                                     "task_id": cohort["task_id"], "init_id": ep["init_id"], "condition": condition,
                                     "model_device": cohort["model_device"], "success": ep["success"],
                                     "environment_steps": ep["steps"], "source_trajectory_init": ep["source_trajectory_init"]})
    print(json.dumps({"results": str(args.out_dir / "results.json"),
                      "cohorts": [{"task_id": c["task_id"], "observed_arms": c["observed_arms"], "missing_arms": c["missing_arms"]} for c in result["cohorts"]]}))


if __name__ == "__main__":
    main()
