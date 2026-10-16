#!/usr/bin/env python3
"""Audit recorded Frozen / NoLangTTT outcomes without loading models or environments.

Only a completed three-init panel produces paired statistics. Historical Task0
is descriptive and is never pooled with the registered Task1 panel. Files in
the output directory are created exclusively; existing evidence is not replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["Frozen", "NoLangTTT"]
SPLIT = "six_action_purge1_disjoint_images_v2"
OPTIMIZER = {"type": "AdamW", "lr": .0005, "weight_decay": .01,
             "loss_scale": .1, "gradient_accumulation": 30, "updates": 1,
             "clip_norm": .1, "clip_after_every_microbatch": True}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read(value):
    return json.loads(path(value).read_text())


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(value):
    return {"path": str(path(value)), "sha256": sha(value)}


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def same(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, label + " differs")


def subset(actual, expected, label):
    for key, value in expected.items():
        same(actual.get(key), value, f"{label}.{key}")


def bound_file(filename, expected):
    require(digest_valid(expected), f"Missing/invalid expected digest: {filename}")
    result = record(filename)
    same(result["sha256"], expected, str(filename) + " SHA256")
    return result


def exact_metric(metric, label):
    same(metric.get("exact"), True, label + ".exact")
    for key in ("max_abs", "l2", "relative_l2"):
        require(type(metric.get(key)) in (int, float) and metric[key] == 0, label + "." + key)


def exact_mcnemar(gains, losses):
    """Two-sided conditional binomial McNemar; no asymptotic approximation."""
    require(type(gains) is int and type(losses) is int and min(gains, losses) >= 0, "Invalid discordances")
    n = gains + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def validate_registration(reg):
    p = reg["protocol"]
    subset(p, {"suite": "libero_10", "task_id": 1, "source_init_id": 0,
               "eval_init_ids": [0, 1, 2], "conditions": CONDITIONS, "adaptation_text": "no lang",
               "max_steps": 600, "train_windows": 30, "heldout_windows": 14,
               "deployment_mask": "native", "split": SPLIT}, "registration")
    subset(p["optimizer"], OPTIMIZER, "registration.optimizer")
    subset(p["lora"], {"rank": 64, "alpha": 32, "dropout": 0}, "registration.lora")
    same(p["train_starts"], list(range(0, 180, 6)), "registered training prefix")
    same(p["purged_starts"], [180], "registered purge")
    same(p["heldout_starts"], list(range(186, 270, 6)), "registered heldout")
    bindings = {k: bound_file(k, v) for k, v in reg["source_file_sha256"].items()}
    source = reg["source"]
    bindings["native_summary"] = bound_file(source["summary_path"], source["summary_sha256"])
    bindings["native_source_trace"] = bound_file(source["trace_path"], source["trace_sha256"])
    old = reg["old_conditional_plan"]
    bindings["historical_plan_identity_only"] = bound_file(old["path"], old["sha256"])
    return bindings


def validate_trace(ep, condition, p):
    subset(ep, {"suite": p["suite"], "task_id": p["task_id"], "condition": condition,
                "seed": p["environment_seed"], "goal": p["clean_goal"],
                "settle_steps": p["settle_steps"], "success_at_start": False}, "episode")
    if condition != "native":
        same(ep.get("model_device"), p["model_device"], "episode device")
    require(type(ep.get("init_id")) is int and ep["init_id"] in [0, 1, 2], "Invalid initial-state ID")
    require(type(ep.get("success")) is bool and type(ep.get("terminal_done")) is bool, "Nonboolean result")
    n = ep.get("environment_steps")
    require(type(n) is int and 1 <= n <= p["max_steps"], "Invalid episode length")
    trace = [json.loads(line) for line in path(ep["trace"]).read_text().splitlines() if line.strip()]
    same(len(trace), n, "trace length")
    for i, row in enumerate(trace, 1):
        same(row.get("step"), i, "trace step")
        same(row.get("chunk"), (i - 1) // 3 + 1, "trace action chunk")
        require(type(row.get("task_success")) is bool and type(row.get("done")) is bool, "Trace nonboolean outcome")
        for field, size in (("action", 7), ("eef_before", 3), ("eef_after", 3),
                            ("eef_quaternion", 4), ("gripper_qpos", 2)):
            values = row.get(field)
            require(isinstance(values, list) and len(values) == size
                    and all(type(x) in (int, float) and math.isfinite(x) for x in values), "Malformed trace " + field)
        require(type(row.get("reward")) in (int, float) and math.isfinite(row["reward"]), "Invalid reward")
        for field in ("condition", "goal"):
            if field in row:
                same(row[field], condition if field == "condition" else p["clean_goal"], "trace " + field)
        require(i == n or not (row["done"] or row["task_success"]), "Execution continued past terminal state")
    same(trace[-1]["task_success"], ep["success"], "trace success")
    same(trace[-1]["done"], ep["terminal_done"], "trace done")
    require(ep["success"] or ep["terminal_done"] or n == p["max_steps"], "Incomplete nonterminal failure")
    same(ep.get("chunks"), (n + 2) // 3, "episode chunks")
    for key in ("initial_state_sha256", "bddl_sha256"):
        require(digest_valid(ep.get(key)), "Missing physical-state/task digest")
    return {"task_id": ep["task_id"], "init_id": ep["init_id"], "condition": condition,
            "success": ep["success"], "executed_steps": n,
            "capped_steps": n if ep["success"] else p["max_steps"],
            "initial_state_sha256": ep["initial_state_sha256"], "bddl_sha256": ep["bddl_sha256"],
            "trace": record(ep["trace"])}


def validate_summary(filename, reg=None, registration_path=None):
    doc = read(filename)
    require(doc.get("status") in ("completed", "stopped_no_lang_ttt_gate_failed"), "Summary is not terminal")
    same(doc.get("conditions"), CONDITIONS, "executed conditions")
    same(set(doc.get("results", {})), set(CONDITIONS), "result conditions")
    same(doc.get("prepare_data_only"), False, "prepare-only flag")
    same(doc.get("api_requests"), 0, "API request count")
    p, data = doc["protocol"], doc["data"]
    subset(p, {"suite": "libero_10", "source_init_id": 0, "eval_init_ids": [0, 1, 2],
               "normal_adaptation_text": "no lang", "normal_adaptation_conditions": CONDITIONS,
               "train_windows": 30, "max_steps": 600}, "protocol")
    require(type(p["task_id"]) is int and p["task_id"] in (0, 1), "Unexpected task")
    require(p["model_device"] == "cuda", "Byte parity requires the registered native CUDA device")
    identity = {k: v for k, v in p.items() if k != "description"}
    identity["cache_sha256"] = data["sha256"]
    same(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
         doc["protocol_identity_sha256"], "protocol/cache identity")
    sources = {k: bound_file(k, v) for k, v in p["source_files"].items()}
    same(p["script_sha256"], p["source_files"]["scripts/pilot_worldagen_cached_metadata_ttt.py"], "runner identity")
    bound_file("models/WorldAgen/libero-38.pth", p["checkpoint_sha256"])
    if reg:
        rp = reg["protocol"]
        mapping = {"seed": "environment_seed", "device": "model_device", "threads": "torch_threads",
                   "policy_goal": "clean_goal", "adaptation_text": "normal_adaptation_text"}
        for key in ("suite", "task_id", "source_init_id", "eval_init_ids", "seed", "adapter_seed", "device",
                    "threads", "camera_size", "orientation", "settle_steps", "max_steps", "train_windows",
                    "heldout_windows", "policy_goal", "adaptation_text"):
            same(p.get(mapping.get(key, key)), rp[key], "registered " + key)
        same(path(filename).parent, path(reg["outputs"]["evaluation_dir"]), "registered result directory")
        same(p["checkpoint_sha256"], reg["source_file_sha256"]["models/WorldAgen/libero-38.pth"], "registered checkpoint")
        for key, value in reg["source_file_sha256"].items():
            if key in p["source_files"]:
                same(p["source_files"][key], value, "registered source " + key)
    subset(p["optimizer"], OPTIMIZER, "optimizer")
    subset(p["lora"], {"rank": 64, "alpha": 32, "dropout": 0,
                       "projections": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}, "lora")
    split = p["data_split"]
    subset(split, {"version": SPLIT, "train_starts": list(range(0, 180, 6)), "purged_starts": [180],
                   "window_action_count": 6, "input_offsets": [0, 3], "target_offsets": [3, 6],
                   "cross_partition_images_disjoint": True}, "split")
    same(split["heldout_starts"], list(range(186, 186 + 6 * p["heldout_windows"], 6)), "heldout ordering")
    for partition in ("train", "heldout"):
        expected = sorted({start + offset for start in split[partition + "_starts"] for offset in (0, 3, 6)})
        same(split[partition + "_frame_indices"], expected, partition + " image indices")
        same(data[partition + "_starts"], split[partition + "_starts"], partition + " cache starts")
    require(not set(split["train_frame_indices"]) & set(split["heldout_frame_indices"]), "Data leakage across split")
    require(max(split["heldout_frame_indices"]) <= split["source_action_count"], "Heldout exceeds source")
    val = data["validation"]
    subset(val, {"schema": 3, "legacy_cache_accepted": False, "source_actions_exact": True,
                 "frame_window_alignment_exact": True, "split_protocol": split}, "cache validation")
    physical = {"suite": p["suite"], "task_id": p["task_id"], "source_init_id": p["source_init_id"],
                "seed": p["environment_seed"], "camera_size": p["camera_size"], "settle_steps": p["settle_steps"],
                "orientation": p["orientation"], "source_max_steps": p["max_steps"], "processed_image_size": 224,
                "env_action_dtype": "float16", "image_rotation_degrees": 0, "patch_whitening": False}
    subset(val["physical_protocol"], physical, "physical data protocol")
    if reg:
        for key in ("train_starts", "heldout_starts", "purged_starts"):
            same(split[key], reg["protocol"][key], "registered " + key)
    cache = bound_file(data["path"], data["sha256"])
    input_binding = None
    if reg:
        binding_path = path(reg["outputs"]["data_dir"]) / "clean_input_binding.json"
        binding = read(binding_path)
        same(binding["registration_sha256"], sha(registration_path), "pre-evaluation registration binding")
        same(path(binding["registration_path"]), path(registration_path), "registration binding path")
        same(binding["conditions"], CONDITIONS, "input binding conditions")
        same(binding["new_task_evaluation_episodes_before_binding"], 0, "input binding time boundary")
        same(binding["physical_cache_sha256"], data["sha256"], "registered prepared-cache digest")
        same(path(binding["physical_cache_path"]), path(data["path"]), "registered prepared-cache path")
        bound_file(binding["prepared_summary_path"], binding["prepared_summary_sha256"])
        prepared = read(binding["prepared_summary_path"])
        same(prepared["status"], "prepared_data_only", "prepared summary status")
        same(prepared["data"]["sha256"], data["sha256"], "prepared data digest")
        same(prepared["data"]["validation"], val, "prepared cache validation")
        input_binding = record(binding_path)
    replay = data["replay"]
    same(replay["checked_steps"], max(split["heldout_frame_indices"]), "replay extent")
    same(replay["reward_done_success_exact"], True, "replay physical outcome parity")
    same(replay["atol"], reg["protocol"]["replay_atol"] if reg else 1e-9, "replay tolerance")
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= replay["atol"]
                for v in replay["max_abs"].values()) and bool(replay["max_abs"]), "Replay state mismatch")
    same(doc["frozen_parameters_unchanged"], True, "frozen-parameter audit")
    before, after = doc["frozen_parameter_sha256_before"], doc["frozen_parameter_sha256_after"]
    require(digest_valid(before) and before == after, "Frozen parameter digests differ")
    zero = doc["zero_adapter_equivalence"]
    subset(zero, {"model_device": p["model_device"], "window_start": 0, "seed": 20261014}, "zero adapter")
    same(set(zero["outputs"]), {"policy_arm", "policy_gripper", "world"}, "zero adapter outputs")
    for key, metric in zero["outputs"].items():
        exact_metric(metric, "zero adapter " + key)
    source = p["source"]
    native_record = bound_file(source["summary_path"], source["summary_sha256"])
    bound_file(source["trace_path"], source["trace_sha256"])
    if reg:
        for key in ("summary_sha256", "trace_sha256"):
            same(source[key], reg["source"][key], "registered source " + key)
        same(split["source_action_count"], reg["source"]["environment_steps"], "registered source length")
    native = read(source["summary_path"])
    same(native["status"], "completed", "native panel status")
    subset(native["protocol"], {"suite": p["suite"], "seed": p["environment_seed"],
           "orientation": p["orientation"], "camera_size": p["camera_size"],
           "max_steps": p["max_steps"], "settle_steps": p["settle_steps"]}, "native physical protocol")
    same(native["checkpoint_provenance"]["sha256"], p["checkpoint_sha256"], "native checkpoint")
    native_rows = [r for r in native["results"] if r["task_id"] == p["task_id"]]
    same(sorted((r["condition"], r["init_id"]) for r in native_rows), [("native", i) for i in range(3)], "native init panel")
    native_by_init = {r["init_id"]: r for r in native_rows}
    same(sha(native_by_init[0]["trace"]), source["trace_sha256"], "source-init native trace")
    bddl_files = p["same_scene_swap_registration"]["bddl_files"]
    bddl = native_rows[0]["bddl_sha256"]
    matching_bddl = [r for r in bddl_files if r["sha256"] == bddl]
    require(len(matching_bddl) == 1, "Task BDDL not bound by runner")
    bddl_record = bound_file(matching_bddl[0]["path"], bddl)
    metadata = doc["cached_metadata"]["no_lang"]
    same(metadata["literal_text"], "no lang", "canonical adaptation text")
    metadata_record = bound_file(metadata["file"], metadata["sha256"])
    initial = doc["initial_adapter_sha256"]
    require(digest_valid(initial), "Invalid initial adapter digest")
    artifacts = {"initial_adapter": record(path(filename).parent / "initial_adapter.pt")}
    episodes, parity, diagnostics = [], [], {}
    for condition in CONDITIONS:
        arm = doc["results"][condition]
        subset(arm, {"condition": condition, "adaptation_text": "no_lang", "adaptation_literal": "no lang",
                     "adaptation_metadata_sha256": metadata["sha256"]}, condition)
        training = arm["training"]
        subset(training, {"optimizer_updates": int(condition == "NoLangTTT"),
                          "microbatches": 30 if condition == "NoLangTTT" else 0}, condition + " training")
        same([r["start"] for r in training["rows"]], split["train_starts"] if condition == "NoLangTTT" else [], "training order")
        for row in training["rows"]:
            require(all(type(row.get(k)) in (float, int) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ("image_mse", "primary_mse", "wrist_mse", "scaled_loss", "pre_clip_accumulated_grad_norm")), "Invalid training metric")
            require(math.isclose(row["scaled_loss"], row["image_mse"] * .1 / 30, rel_tol=1e-6, abs_tol=1e-12), "Training loss scale")
        if condition == "Frozen":
            same(arm["adapter_sha256"], initial, "Frozen adapter identity")
            exact_metric(arm["adapter_change"], "Frozen adapter change")
        else:
            require(digest_valid(training.get("gradient_sha256")), "Missing gradient digest")
        for key in ("adapter_file", "gradient_file"):
            artifacts[condition + "_" + key] = record(path(filename).parent / arm[key])
        same([e["init_id"] for e in arm["episodes"]], [0, 1, 2], "complete ordered three-init panel")
        for ep in arm["episodes"]:
            checked = validate_trace(ep, condition, p)
            reference = native_by_init[ep["init_id"]]
            for key in ("initial_state_sha256", "bddl_sha256", "task_name", "goal"):
                same(ep[key], reference[key], "native physical task/state " + key)
            if condition == "Frozen":
                native_checked = validate_trace(reference, "native", p)
                exact = path(ep["trace"]).read_bytes() == path(reference["trace"]).read_bytes()
                require(exact, f"Frozen trace is not byte-exact with native init{ep['init_id']}")
                parity.append({"init_id": ep["init_id"], "byte_exact": exact,
                               "Frozen_trace": checked["trace"], "native_trace": native_checked["trace"],
                               "excluded_fields": [], "scope": "Entire JSONL bytes, not selected fields"})
            episodes.append(checked)
        successes = sum(e["success"] for e in arm["episodes"])
        same(arm["task_successes"], successes, "arm success count")
        same(arm["task_success_rate"], successes / 3, "arm success rate")
        diagnostics[condition] = {}
        for key in ("heldout_before_adaptation_condition", "heldout_after_adaptation_condition",
                    "heldout_after_original_goal_native", "heldout_after_no_lang_native"):
            metric = arm[key]
            same(metric["windows"], p["heldout_windows"], "heldout count")
            same([r["start"] for r in metric["rows"]], split["heldout_starts"], "heldout ordering")
            values = [r["image_mse"] for r in metric["rows"]]
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), "Invalid forecast metric")
            require(math.isclose(metric["image_mse_mean"], statistics.mean(values), rel_tol=1e-7, abs_tol=1e-12), "Forecast summary mismatch")
            diagnostics[condition][key] = metric["image_mse_mean"]
    return {"summary": record(filename), "task_id": p["task_id"], "source_init_id": p["source_init_id"],
            "protocol_identity_sha256": doc["protocol_identity_sha256"], "episodes": episodes,
            "Frozen_native_trace_parity": parity, "physical_cache": cache, "source_native_summary": native_record,
            "source_files": sources, "bddl": bddl_record, "canonical_metadata": metadata_record,
            "artifacts": artifacts, "pre_evaluation_input_binding": input_binding,
            "frozen_parameter_sha256": before, "zero_adapter_equivalence": zero,
            "forecast_diagnostics_not_episodes": {"heldout_windows": p["heldout_windows"],
                "same_source_trajectory": True, "no_cross_partition_images": True, "means": diagnostics},
            "integrity_scope": "Source/cache/artifact files hashed; runtime-reported tensor/output equality checked. No tensors or models re-executed by this aggregator."}


def paired_statistics(episodes, inferential=True):
    arms = {c: {e["init_id"]: e for e in episodes if e["condition"] == c} for c in CONDITIONS}
    require(all(set(a) == {0, 1, 2} for a in arms.values()) and len(episodes) == 6, "Incomplete paired panel")
    rows = []
    for init in range(3):
        a, b = arms["Frozen"][init], arms["NoLangTTT"][init]
        for key in ("task_id", "initial_state_sha256", "bddl_sha256"):
            same(a[key], b[key], "paired " + key)
        rows.append({"task_id": a["task_id"], "init_id": init, "source_init_development": init == 0,
            "Frozen_success": a["success"], "NoLangTTT_success": b["success"],
            "Frozen_executed_steps": a["executed_steps"], "NoLangTTT_executed_steps": b["executed_steps"],
            "Frozen_capped_steps": a["capped_steps"], "NoLangTTT_capped_steps": b["capped_steps"],
            "success_difference": int(b["success"]) - int(a["success"]),
            "capped_steps_difference": b["capped_steps"] - a["capped_steps"],
            "both_success_steps_difference": b["executed_steps"] - a["executed_steps"] if a["success"] and b["success"] else None})
    gained = [r["init_id"] for r in rows if r["success_difference"] == 1]
    lost = [r["init_id"] for r in rows if r["success_difference"] == -1]
    counts = {c: sum(e["success"] for e in arms[c].values()) for c in CONDITIONS}
    groups = {}
    for label, ids in (("source_init0_development", [0]), ("non_source_init1_2", [1, 2])):
        selected = [r for r in rows if r["init_id"] in ids]
        groups[label] = {"init_ids": ids, "episodes_per_condition": len(ids),
            "successes": {c: sum(r[c + "_success"] for r in selected) for c in CONDITIONS},
            "capped_steps_total": {c: sum(r[c + "_capped_steps"] for r in selected) for c in CONDITIONS},
            "capped_steps_difference_total": sum(r["capped_steps_difference"] for r in selected),
            "capped_steps_difference_mean": statistics.mean(r["capped_steps_difference"] for r in selected),
            "inferential_statistics": None,
            "scope": "Adaptation-source development case" if ids == [0] else "Initial states supplying no adaptation data; one shared adapted model, same task, descriptive only"}
    return {"episodes_per_condition": 3, "successes": counts,
            "success_rates": {c: counts[c] / 3 for c in CONDITIONS},
            "gained_init_ids": gained, "lost_init_ids": lost,
            "success_rate_difference_NoLangTTT_minus_Frozen": (len(gained) - len(lost)) / 3,
            "capped_steps_total": {c: sum(e["capped_steps"] for e in arms[c].values()) for c in CONDITIONS},
            "mean_capped_steps_difference_NoLangTTT_minus_Frozen": statistics.mean(r["capped_steps_difference"] for r in rows),
            "normal_health_gate": counts["Frozen"] > 0 and not lost,
            "positive_baseline_benefit_descriptive": bool(gained) and not lost,
            "exact_McNemar_two_sided": {"discordant_pairs": len(gained) + len(lost),
                "p_value": exact_mcnemar(len(gained), len(lost)),
                "method": "2*Binomial(n=gains+losses,p=0.5) lower-tail at min(gains,losses), capped at 1; zero discordances p=1",
                "caveat": "Conditional on exchange symmetry of discordant pairs. Three selected initial states of one task sharing one adaptation trajectory; this descriptive, low-power calculation is not a random-population or cross-task robustness inference."} if inferential else None,
            "source_init0": rows[0], "non_source_init1_2": rows[1:], "paired_rows": rows,
            "descriptive_subgroups": groups,
            "confidence_intervals": None, "heldout_windows_used_as_sample_size": False}


def analyze(registration, summary, reference=None):
    reg = read(registration)
    bindings = validate_registration(reg)
    result = {"status": "incomplete", "scope": "Normal Frozen versus literal-no-lang adaptation only",
              "aggregator": record(__file__), "registration": record(registration),
              "registered_sources": bindings, "primary": None, "historical_reference": None,
              "statistical_limits": "No pseudo-independent window CI; Task0 never pooled with Task1. Source init0 is development data; init1/2 share the same adapted model. Task choice and native source success were known in advance."}
    if not path(summary).exists():
        result["reason"] = "Registered summary not yet present; no outcomes imputed"
        return result
    doc = read(summary)
    require(set(doc.get("conditions", [])) <= set(CONDITIONS)
            and set(doc.get("results", {})) <= set(CONDITIONS), "Non-clean condition in input")
    if doc.get("status") not in ("completed", "stopped_no_lang_ttt_gate_failed"):
        result["summary"] = record(summary)
        result["reason"] = "Nonterminal or resource/error run; no paired verdict or p-value"
        result["runner_status"] = doc.get("status")
        return result
    primary = validate_summary(summary, reg, registration)
    primary["statistics"] = paired_statistics(primary["episodes"])
    result["primary"] = primary
    if reference:
        historical = validate_summary(reference)
        same(historical["task_id"], 0, "historical reference task")
        historical["statistics"] = paired_statistics(historical["episodes"], inferential=False)
        historical["scope"] = "Previously observed selected Task0 development result; descriptive historical comparison only, not an independent validation sample or pooled statistical test."
        result["historical_reference"] = historical
    result["status"] = "completed_validated_clean_panel"
    return result


def self_test():
    assert exact_mcnemar(0, 0) == 1 and exact_mcnemar(2, 0) == .5 and exact_mcnemar(3, 0) == .25
    assert exact_mcnemar(0, 3) == .25 and exact_mcnemar(1, 1) == 1
    eps = [{"task_id": 1, "init_id": i, "condition": c, "success": success,
            "executed_steps": steps, "capped_steps": steps if success else 600,
            "initial_state_sha256": str(i), "bddl_sha256": "same"}
           for c, values in (("Frozen", [(True, 275), (False, 600), (False, 123)]),
                             ("NoLangTTT", [(True, 300), (True, 350), (False, 600)]))
           for i, (success, steps) in enumerate(values)]
    s = paired_statistics(eps)
    assert s["gained_init_ids"] == [1] and not s["lost_init_ids"]
    assert s["capped_steps_total"]["Frozen"] == 1475
    assert s["mean_capped_steps_difference_NoLangTTT_minus_Frozen"] == -75
    assert paired_statistics(eps, False)["exact_McNemar_two_sided"] is None
    try:
        paired_statistics(eps[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete panel accepted")
    print(json.dumps({"self_test": "passed", "scope": "Pure CPU arithmetic/schema; no environment/model execution"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.registration, args.summary, args.out_dir)):
        parser.error("--registration, --summary and --out-dir are required")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("analysis.json", "paired_results.csv", "condition_results.csv")]
    if any(p.exists() for p in targets):
        parser.error("Output evidence exists; select a new --out-dir")
    try:
        result = analyze(args.registration, args.summary, args.reference_summary)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "invalid", "error": {"type": type(error).__name__, "message": str(error)},
                  "aggregator": record(__file__), "primary": None, "historical_reference": None,
                  "paired_verdict": None, "statistics": None}
    with targets[0].open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    paired, condition_rows = [], []
    for scope in ("primary", "historical_reference"):
        cohort = result.get(scope)
        if not cohort:
            continue
        paired.extend({"scope": scope, **row} for row in cohort["statistics"]["paired_rows"])
        for group, ids in (("all_three", [0, 1, 2]), ("source_init0_development", [0]), ("non_source_init1_2", [1, 2])):
            for c in CONDITIONS:
                eps = [e for e in cohort["episodes"] if e["condition"] == c and e["init_id"] in ids]
                condition_rows.append({"scope": scope, "task_id": cohort["task_id"], "group": group, "condition": c,
                    "episodes": len(eps), "successes": sum(e["success"] for e in eps),
                    "success_rate": sum(e["success"] for e in eps) / len(eps),
                    "total_executed_steps": sum(e["executed_steps"] for e in eps),
                    "total_capped_steps": sum(e["capped_steps"] for e in eps)})
    for target, rows, empty_fields in ((targets[1], paired, ["scope", "task_id", "init_id"]),
                                       (targets[2], condition_rows, ["scope", "task_id", "condition"])):
        with target.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": result["status"], "analysis": str(targets[0]), "error": result.get("error")}))
    sys.exit(0 if result["status"] == "completed_validated_clean_panel" else 2)


if __name__ == "__main__":
    main()
