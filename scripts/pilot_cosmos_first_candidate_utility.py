#!/usr/bin/env python3
"""Clean-only first-query candidate utility under the unchanged native N=2 continuation.

No model/environment is imported by preparation, preflight, or self-test. Execution
reuses native.run_episode and regenerates the first query before changing only its
selected index. Existing Clean evidence is the only scientific input panel.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import aggregate_cosmos_native_clean_gate as audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/summary.json"
SOURCE_SHA = "075fe874619b45e9f1dde03ceafcfee238063e06f97ad104bd9cd97f93b0b50a"
DEFAULT_CONFIG = ROOT / "configs/cosmos_first_candidate_utility_matched30_v1.json"
DEFAULT_OUT = ROOT / "runs/cosmos_first_candidate_utility_matched30_v1"
SCOPE = "clean_native_first_query_ranking_contribution"
PROTOCOL = {
    "suite": "libero_goal", "task_ids": list(range(10)), "init_ids": [0, 1, 2],
    "candidate_indices": [0, 1], "seed": 31, "env_seed": 0, "settle_steps": 10,
    "max_steps": 300, "best_of_n": 2, "camera_size": 256,
    "candidate_seeds": [31, 32], "chunk_length": 16, "offload": False,
    "cuda_memory_fraction": 0.28, "video_stride": 2,
    "continuation": "original task Goal, native N=2 argmax at every query after the first",
    "stopping": "unchanged native stop immediately on environment success or done; otherwise 300 actions",
    "execution_order": "task ascending, init ascending, original Clean selected candidate then the other candidate",
    "seed_rule": "native run_episode resets seed31 independently for every fresh branch; order never enters seeds",
}
RESOURCE_GATE = {
    "primary": "mean(success_native_argmax - (success_candidate0 + success_candidate1)/2)",
    "minimum_native_minus_alternate_net_successes": 3,
    "minimum_native_minus_uniform_success_rate": 0.05,
    "minimum_tasks_with_positive_native_minus_alternate_net_successes": 2,
    "secondary": "failure-penalized completion cost: successful steps, otherwise 300; cannot rescue primary gate",
    "controls": ["fixed_candidate0", "fixed_candidate1", "uniform_expectation", "two_candidate_hindsight_oracle"],
    "incomplete": "No verdict until all60 valid and all30 original-selected trajectories pass exact parity",
    "uncertainty": "Exploratory task-cluster summaries; 10 tasks, not 30 independent tasks",
    "stop_rule": "If the fixed gate fails, do not select another query, continuation or horizon to search for a positive result",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return audit.sha(Path(path))


def read(path):
    return audit.read_json(Path(path))


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(data)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(data)
        temporary.replace(path)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_clean_panel():
    source, source_sha = read(SOURCE)
    require(source_sha == SOURCE_SHA, "Fixed Clean source summary changed")
    require(source["status"] == "completed" and source["condition"] == "clean", "Incomplete/non-Clean source")
    for key in ("suite", "task_ids", "init_ids", "seed", "env_seed", "settle_steps", "max_steps", "best_of_n", "offload", "cuda_memory_fraction"):
        require(source["protocol"][key] == PROTOCOL[key], f"Clean protocol changed: {key}")
    gate, gate_sha = read(audit.GATE)
    require(gate_sha == audit.GATE_SHA, "Original physical protocol gate changed")
    require(len(source["results"]) == 30, "Exactly 30 Clean source episodes required")
    rows = {}
    references = {rel(SOURCE): source_sha, rel(audit.GATE): gate_sha}
    for row in source["results"]:
        key = row["task_id"], row["init_id"]
        require(key not in rows and key in {(t, i) for t in range(10) for i in range(3)}, "Duplicate/unregistered Clean episode")
        require(row["provenance"] == source["provenance"], "Source episode provenance differs from summary")
        checked = audit.audit_episode(row, SOURCE, gate)
        ep_path = Path(row["trace"]).with_suffix(".json")
        ep, ep_sha = read(ep_path)
        require(ep == row, "Source episode JSON differs from summary")
        references[rel(ep_path)] = ep_sha
        for item in checked["artifacts"].values():
            references[rel(item["path"])] = item["sha256"]
        chunks = audit.records(row["chunks"])
        first = chunks[0]
        require(first["first_step"] == 1 and first["chunk"] == 1, "Source query is not the fixed first query")
        rows[key] = {"episode": row, "episode_json": str(ep_path.resolve()), "first_chunk": first}
    require(set(rows) == {(t, i) for t in range(10) for i in range(3)}, "Incomplete Clean grid")
    return source, rows, references


def source_inventory(source):
    original = source["provenance"]
    for name, expected in original["source_sha256"].items():
        require(sha(ROOT / name) == expected, f"Original Clean computation source changed: {name}")
    require(sha(ROOT / "scripts/cosmos_policy_native_inference.py") == original["helper_sha256"], "Original inference helper changed")
    require(sha(ROOT / "scripts/pilot_cosmos_policy_native_task_gate.py") == original["runner_sha256"], "Original native runner changed")
    paths = {ROOT / name for name in source["provenance"]["source_sha256"]}
    paths.update(Path(__file__).parent / name for name in (
        "pilot_cosmos_first_candidate_utility.py", "pilot_cosmos_policy_native_task_gate.py",
        "cosmos_policy_native_inference.py", "aggregate_cosmos_native_clean_gate.py",
        "aggregate_cosmos_first_candidate_utility.py"))
    for directory in (ROOT / "vendor/LIBERO/libero/libero", ROOT / "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos"):
        paths.update(directory.rglob("*.py"))
    return {rel(p): sha(p) for p in sorted(paths)}


def physical_inventory(rows):
    files = {}
    for item in rows.values():
        row = item["episode"]
        bddl = Path(row["bddl"])
        init = ROOT / "vendor/LIBERO/libero/libero/init_files" / bddl.parent.name / (bddl.stem + ".pruned_init")
        require(sha(bddl) == row["bddl_sha256"] and sha(init) == row["init_file_sha256"], "Physical definition/init changed")
        files[rel(bddl)], files[rel(init)] = sha(bddl), sha(init)
    return files


def model_inventory(source):
    p = source["provenance"]
    return {
        p["checkpoint"]: p["checkpoint_sha256"], p["tokenizer"]: p["tokenizer_sha256"],
        str(Path(p["checkpoint"]).parent / "libero_t5_embeddings.pkl"): p["text_embeddings_sha256"],
        str(Path(p["checkpoint"]).parent / "libero_dataset_statistics.json"): p["stats_sha256"],
    }


def schedule(rows):
    result = []
    for task, init in sorted(rows):
        selected = rows[(task, init)]["first_chunk"]["native_best_of_n"]["selected_index"]
        require(type(selected) is int and selected in (0, 1), "Invalid Clean selection")
        for candidate in (selected, 1 - selected):
            result.append({"task_id": task, "init_id": init, "forced_candidate_index": candidate,
                           "source_native_selected_index": selected,
                           "role": "native_selected" if candidate == selected else "alternate"})
    return result


def prepare(path, out_dir):
    source, rows, references = load_clean_panel()
    config = {
        "schema_version": 1, "scope": SCOPE, "status": "prospective_clean_only_registration",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": binding(SOURCE), "out_dir": str(Path(out_dir).resolve()),
        "runner": binding(__file__), "protocol": PROTOCOL, "resource_gate": RESOURCE_GATE,
        "source_sha256": source_inventory(source), "reference_sha256": references,
        "physical_file_sha256": physical_inventory(rows), "model_sha256": model_inventory(source),
        "planned_branches": schedule(rows),
        "known_evidence": "All30 original-selected Clean outcomes are already known. Alternate first-query outcomes are the new measurements. This is development evaluation, not a blind test.",
        "scope_boundaries": [
            "Only original Clean source episodes are consumed. No alternate Goal selection or routing exists.",
            "Both candidates are regenerated with unchanged native forward/RNG before selecting the registered candidate.",
            "The policy continuation rule is shared; later controls depend normally on each branch's resulting observations.",
            "The first selection changes only once. This does not estimate every-query ranking contribution or a pure forecast-only effect.",
            "Failure of any previous scientific gate remains in force; this normal baseline neither activates nor revises it.",
            "Native inner condition=clean denotes original Goal routing; outer forced_candidate_index truthfully records the first-selection intervention.",
        ],
        "parity_policy": "Before the alternate branch, the selected branch must reproduce every recorded physical trace field, all candidate values/controls/selections, state dtype/shape/bytes and initial/final PNG exactly. Both branches require initial/settled state, initial PNG and regenerated first candidates exact. Any mismatch is terminal; preserve logs, no silent retry.",
        "runtime_requirements": {"MUJOCO_GL": "egl", "ALPHABRAIN_USE_NATIVE_TE": "not 1",
                                 "torch_version": source["provenance"]["torch_version"],
                                 "numpy_version": source["provenance"]["numpy_version"],
                                 "offload": False, "cudnn_benchmark": False, "cudnn_deterministic": True},
        "api_requests": 0, "training_updates": 0,
    }
    write(path, config, exclusive=True)
    return binding(path)


def preflight(path):
    config, config_sha = read(path)
    require(config.get("scope") == SCOPE and config.get("schema_version") == 1, "Wrong registration scope")
    require(config.get("status") == "prospective_clean_only_registration", "Wrong registration status")
    require(config["protocol"] == PROTOCOL and config["resource_gate"] == RESOURCE_GATE, "Protocol/gate changed")
    require(config["api_requests"] == config["training_updates"] == 0, "Unexpected API/training")
    require(config["runner"] == binding(__file__), "Runner hash/path changed after registration")
    require(config["source_summary"] == {"path": str(SOURCE), "sha256": SOURCE_SHA}, "Unregistered source")
    source, rows, references = load_clean_panel()
    require(config["planned_branches"] == schedule(rows), "Registered branch schedule changed")
    require(config["reference_sha256"] == references, "Source artifact catalog changed")
    require(config["source_sha256"] == source_inventory(source), "Source catalog/hash changed")
    require(config["physical_file_sha256"] == physical_inventory(rows), "Physical input catalog changed")
    require(config["model_sha256"] == model_inventory(source), "Model catalog changed")
    for mapping in (config["source_sha256"], config["reference_sha256"], config["physical_file_sha256"], config["model_sha256"]):
        for name, expected in mapping.items():
            require(sha(ROOT / name) == expected, f"Artifact hash mismatch: {name}")
    return config, source, rows, {"status": "cpu_preflight_passed", "config": {"path": str(Path(path).resolve()), "sha256": config_sha},
                                "runner_sha256": sha(__file__), "source_episodes": len(rows), "planned_branches": 60,
                                "source_files": len(config["source_sha256"]), "references": len(references),
                                "gpu_used": False, "environment_created": False, "api_requests": 0, "training_updates": 0}


def compare_first(selection, source_chunk, forced):
    expected = source_chunk["native_best_of_n"]
    actual = selection["candidates"]
    checks = {"original_selected_index": selection["selected_index"] == expected["selected_index"],
              "candidate_count": len(actual) == len(expected["candidates"]) == 2}
    for index, (got, old) in enumerate(zip(actual, expected["candidates"])):
        checks[f"candidate{index}_index_seed"] = (got["index"] == old["index"] == index and got["seed"] == old["seed"] == 31 + index)
        checks[f"candidate{index}_controls_dtype_shape_bytes"] = audit.exact_array(got["actions"][0], np.asarray(old["environment_actions"], dtype=np.float64))
        checks[f"candidate{index}_value_exact"] = audit.exact_array(np.asarray(got["value"], dtype=np.float64), np.asarray(old["predicted_value"], dtype=np.float64))
    old_selected = actual[expected["selected_index"]]
    checks["original_selected_normalized_exact"] = audit.exact_array(
        np.asarray(old_selected["normalized_actions"].float().cpu().numpy()[0].tolist(), dtype=np.float64),
        np.asarray(source_chunk["normalized_actions"], dtype=np.float64))
    rms = old_selected["future_latent"][:, :, 5:8].float().square().mean().sqrt().item()
    checks["original_selected_future_rms_exact"] = audit.exact_array(np.asarray(rms, dtype=np.float64), np.asarray(source_chunk["future_slot_rms"], dtype=np.float64))
    result = {"all_exact": all(checks.values()), "checks": checks,
              "source_native_selected_index": expected["selected_index"], "forced_candidate_index": forced,
              "scope": "Both candidate continuous controls and values; original-selected normalized actions and future RMS. No stored full-latent reference exists."}
    return result


@contextmanager
def first_query_override(inference, env_module, source_item, forced, directory):
    """Sequential, scoped hooks; returned environment state and model tensors unchanged."""
    original_best, original_env = inference.best_of_n, env_module.OffScreenRenderEnv
    source_row = source_item["episode"]
    trace = audit.records(source_row["trace"])
    states_path = Path(source_row["trace"]).with_name(Path(source_row["trace"]).stem + "_states.npz")
    with np.load(states_path, allow_pickle=False) as archive:
        expected_states = {key: archive[key].copy() for key in ("initial", "settled")}
    tracker = {"query_count": 0, "env_count": 0, "state_reads": 0, "checks": {}, "first_query": None}

    def factory(*args, **kwargs):
        require(tracker["env_count"] == 0, "Unexpected additional environment")
        tracker["env_count"] += 1
        env = original_env(*args, **kwargs)
        get_state = env.get_sim_state

        def checked_state():
            value = get_state()
            count = tracker["state_reads"]
            tracker["state_reads"] += 1
            if count < 2:
                key = ("initial", "settled")[count]
                tracker["checks"][key + "_dtype_shape_bytes"] = audit.exact_array(np.asarray(value), expected_states[key])
                write(directory / "initial_parity.json", {"status": "checking", "checks": tracker["checks"]})
                require(tracker["checks"][key + "_dtype_shape_bytes"], f"Initial physical parity failed: {key}")
            return value

        env.get_sim_state = checked_state
        return env

    def best(*args, **kwargs):
        tracker["query_count"] += 1
        if tracker["query_count"] == 1:
            require(tracker["state_reads"] == 2, "Initial/settled state validation did not precede inference")
            initial_image = directory / Path(source_row["initial_image"]).name
            tracker["checks"]["initial_png_bytes"] = sha(initial_image) == sha(source_row["initial_image"])
            obs = args[4]
            tracker["checks"]["first_eef_position"] = audit.exact_array(np.asarray(obs["robot0_eef_pos"]), np.asarray(trace[0]["eef_before"], dtype=np.float64))
            initial = {"all_exact": all(tracker["checks"].values()), "checks": tracker["checks"],
                       "source_state_file": binding(states_path), "source_initial_image": binding(source_row["initial_image"]),
                       "actual_initial_image": binding(initial_image)}
            write(directory / "initial_parity.json", initial)
            require(initial["all_exact"], "Initial observation parity failed")
        selection = original_best(*args, **kwargs)
        if tracker["query_count"] == 1:
            parity = compare_first(selection, source_item["first_chunk"], forced)
            parity["source_chunks"] = binding(source_row["chunks"])
            tracker["first_query"] = parity
            write(directory / "first_query_parity.json", parity)
            require(parity["all_exact"], "First query regeneration differs from registered Clean candidates")
            selection["selected_index"] = forced
        return selection

    inference.best_of_n, env_module.OffScreenRenderEnv = best, factory
    try:
        yield tracker
    finally:
        inference.best_of_n, env_module.OffScreenRenderEnv = original_best, original_env


def audit_branch(row, source_item, forced, directory):
    """Validate truthful forced-first choice plus ordinary native continuation."""
    source = source_item["episode"]
    for key in ("status", "condition", "suite", "best_of_n", "task_id", "init_id", "task_name", "goal", "max_steps", "settle_steps", "seed", "environment_seed", "initial_state_sha256", "bddl", "bddl_sha256", "init_file_sha256", "success_at_start"):
        require(row[key] == source[key], f"Branch physical/Goal protocol differs: {key}")
    steps = row["environment_steps"]
    trace, chunks = audit.records(row["trace"]), audit.records(row["chunks"])
    require(type(steps) is int and 0 < steps <= 300 and len(trace) == steps, "Invalid branch trace length")
    require(len(chunks) == row["action_chunks"] == math.ceil(steps / 16), "Invalid branch chunks")
    require([event["step"] for event in trace] == list(range(1, steps + 1)), "Invalid step order")
    for index, chunk in enumerate(chunks):
        require(chunk["chunk"] == index + 1 and chunk["first_step"] == index * 16 + 1, "Invalid query order")
        sel = chunk["native_best_of_n"]
        candidates = sel["candidates"]
        require(len(candidates) == 2 and [c["index"] for c in candidates] == [0, 1]
                and [c["seed"] for c in candidates] == [31, 32], "Candidate set/seeds changed")
        values = [c["predicted_value"] for c in candidates]
        require(all(math.isfinite(v) for v in values), "Nonfinite value")
        expected = forced if index == 0 else max(range(2), key=lambda i: values[i])
        require(type(sel["selected_index"]) is int and sel["selected_index"] == expected, "Incorrect selection/continuation")
        actions = np.asarray(chunk["environment_actions"], dtype=np.float64)
        require(actions.shape == (16, 7) and np.isfinite(actions).all(), "Malformed controls")
        require(audit.exact_array(actions, np.asarray(candidates[expected]["environment_actions"], dtype=np.float64)), "Executed different candidate")
        if index == 0:
            require(candidates == source_item["first_chunk"]["native_best_of_n"]["candidates"], "First candidates changed")
        for offset, event in enumerate(trace[index * 16:index * 16 + 16]):
            require(event["chunk"] == index + 1 and audit.exact_array(np.asarray(event["action"], dtype=np.float64), actions[offset]), "Trace/control binding failed")
    for index, event in enumerate(trace):
        require(type(event["task_success"]) is bool and type(event["done"]) is bool and math.isfinite(event["reward"]), "Invalid physical outcome")
        if index < steps - 1:
            require(not event["task_success"] and not event["done"], "Continued past native stopping point")
        if index:
            require(event["eef_before"] == trace[index - 1]["eef_after"], "EEF discontinuity")
    require(row["success"] == trace[-1]["task_success"] and row["terminal_done"] == trace[-1]["done"], "Incorrect task utility")
    require(row["success"] or row["terminal_done"] or steps == 300, "Stopped early")
    require(row["reward_sum"] == sum(e["reward"] for e in trace) and row["final_reward"] == trace[-1]["reward"], "Reward mismatch")
    paths = [Path(row["trace"]), Path(row["chunks"]), Path(row["initial_image"]), Path(row["final_image"])]
    states_path = paths[0].with_name(paths[0].stem + "_states.npz")
    with np.load(states_path, allow_pickle=False) as current, np.load(Path(source["trace"]).with_name(Path(source["trace"]).stem + "_states.npz"), allow_pickle=False) as previous:
        require(set(current.files) == {"initial", "settled", "final", "eef_path"}, "State keys changed")
        for key in ("initial", "settled"):
            require(audit.exact_array(current[key], previous[key]), "Archived initial state mismatch")
        require(audit.exact_array(current["eef_path"], np.asarray([e["eef_after"] for e in trace], dtype=np.float64)), "Archived EEF differs")
    require(sha(row["initial_image"]) == sha(source["initial_image"]), "Archived initial PNG differs")
    paths.append(states_path)
    return {p.name: binding(p) for p in paths}


def run(config_path):
    config, source, rows, validation = preflight(config_path)
    output = Path(config["out_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Output must be empty; no resume or silent restart")
    write(output / "cpu_preflight.json", validation, exclusive=True)
    import torch
    import pilot_cosmos_policy_native_task_gate as native
    inference = native.inference
    from libero.libero import benchmark, envs
    require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1" and os.environ.get("MUJOCO_GL") == "egl", "Runtime backend differs")
    require(torch.__version__ == source["provenance"]["torch_version"] and np.__version__ == source["provenance"]["numpy_version"], "Torch/numpy versions differ")
    require(torch.cuda.is_available(), "CUDA required for execution")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(PROTOCOL["cuda_memory_fraction"])
    provenance = copy.deepcopy(source["provenance"])
    provenance["first_candidate_utility_wrapper"] = {"config": validation["config"], "runner_sha256": sha(__file__),
        "scope": SCOPE, "first_selection": "registered original candidate index; native original Goal N2 continuation thereafter"}
    summary = {"schema_version": 1, "status": "loading", "condition": "first_candidate_utility", "scope": SCOPE,
               "config": validation["config"], "runner_sha256": sha(__file__), "source_summary": config["source_summary"],
               "protocol": PROTOCOL, "resource_gate": RESOURCE_GATE, "planned_branches": config["planned_branches"],
               "provenance": provenance, "results": [], "api_requests": 0, "training_updates": 0}
    summary_path = output / "summary.json"
    write(summary_path, summary)
    started = time.monotonic()
    try:
        with (inference.MODEL.parent / "libero_t5_embeddings.pkl").open("rb") as handle:
            text = pickle.load(handle)
        stats = inference.load_stats(inference.MODEL.parent / "libero_dataset_statistics.json")
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        model, vae, sampler, rand = inference.load_model(device="cuda")
        from cosmos_policy_torch_compat import compatibility_metadata
        require(compatibility_metadata() == source["provenance"]["runtime_compatibility"], "Runtime compatibility metadata differs")
        summary["status"] = "running"
        for item in config["planned_branches"]:
            key = item["task_id"], item["init_id"]
            source_item = rows[key]
            forced = item["forced_candidate_index"]
            directory = output / f"task{key[0]:02d}_init{key[1]:02d}" / f"candidate{forced}"
            directory.mkdir(parents=True, exist_ok=False)
            summary["active_branch"] = item
            write(summary_path, summary)
            task = suite.get_task(key[0])
            require(task.language == source_item["episode"]["goal"] and task.name == source_item["episode"]["task_name"], "Original task mapping changed")
            args = SimpleNamespace(**{k: PROTOCOL[k] for k in ("suite", "seed", "env_seed", "settle_steps", "max_steps", "best_of_n", "video_stride")}, out_dir=directory)
            with first_query_override(inference, envs, source_item, forced, directory) as tracker:
                episode = native.run_episode(model, vae, sampler, rand, task, key[0], key[1], text, stats, args, provenance)
            require(tracker["first_query"] is not None and tracker["first_query"]["all_exact"], "Missing first-query validation")
            artifacts = audit_branch(episode, source_item, forced, directory)
            selected_parity = None
            if item["role"] == "native_selected":
                gate, _ = read(audit.GATE)
                audit.audit_episode(episode, summary_path, gate)
                selected_parity = audit.paired_compare(source_item["episode"], episode, SOURCE, summary_path)
                selected_parity["trace_file_bytes_exact"] = sha(source_item["episode"]["trace"]) == sha(episode["trace"])
                selected_parity["all_exact"] &= selected_parity["trace_file_bytes_exact"]
                write(directory / "selected_arm_parity.json", selected_parity)
                require(selected_parity["all_exact"], "Original-selected full trajectory parity failed; alternate branch forbidden")
            first_query_path = directory / "first_query_parity.json"
            initial_parity, _ = read(directory / "initial_parity.json")
            result = {**item, "condition": "first_candidate_utility", "episode": episode,
                      "episode_json": binding(Path(episode["trace"]).with_suffix(".json")),
                      "source_episode_json": binding(source_item["episode_json"]),
                      "first_query_parity": {**binding(first_query_path), "all_exact": True},
                      "initial_parity": initial_parity, "selected_arm_parity": selected_parity,
                      "artifacts": artifacts}
            summary["results"].append(result)
            summary["elapsed_seconds"] = time.monotonic() - started
            write(summary_path, summary)
            print(json.dumps({"completed_branches": len(summary["results"]), **item,
                              "success": episode["success"], "steps": episode["environment_steps"]}), flush=True)
        require(len(summary["results"]) == 60, "Incomplete branch panel")
        summary["status"] = "completed"
        summary.pop("active_branch", None)
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write(summary_path, summary)


def self_test():
    rows = {(t, i): {"first_chunk": {"native_best_of_n": {"selected_index": (t + i) % 2}}} for t in range(10) for i in range(3)}
    plan = schedule(rows)
    checks = {
        "all60_unique": len(plan) == len({(x["task_id"], x["init_id"], x["forced_candidate_index"]) for x in plan}) == 60,
        "selected_then_alternate": all(plan[i]["role"] == "native_selected" and plan[i + 1]["role"] == "alternate" for i in range(0, 60, 2)),
        "same_seeds_all_branches": PROTOCOL["candidate_seeds"] == [31, 32] and PROTOCOL["seed"] == 31,
        "same_original_goal_continuation": PROTOCOL["best_of_n"] == 2 and "original task Goal" in PROTOCOL["continuation"],
        "primary_threshold_exact": RESOURCE_GATE["minimum_native_minus_alternate_net_successes"] / 60 == 0.05,
        "cost_cannot_rescue": "cannot rescue" in RESOURCE_GATE["secondary"],
        "dtype_required": not audit.exact_array(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64)),
        "signed_zero_required": not audit.exact_array(np.array([0.]), np.array([-0.])),
        "partial_no_verdict": "all60" in RESOURCE_GATE["incomplete"],
        "model_not_imported": "torch" not in sys.modules and "libero.libero.envs" not in sys.modules,
    }
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def __getitem__(self, key): return Tensor(self.value[key])
        def float(self): return Tensor(self.value.astype(np.float32))
        def cpu(self): return self
        def numpy(self): return self.value
        def square(self): return Tensor(self.value ** 2)
        def mean(self): return Tensor(self.value.mean())
        def sqrt(self): return Tensor(np.sqrt(self.value))
        def item(self): return self.value.item()
    controls = [np.full((1, 16, 7), index + 0.125, dtype=np.float64) for index in range(2)]
    values = [0.25, 0.75]
    def proposal(*args, **kwargs):
        return {"selected_index": 1, "candidates": [
            {"index": i, "seed": 31 + i, "value": values[i], "actions": controls[i].copy(),
             "normalized_actions": Tensor(controls[i]), "future_latent": Tensor(np.ones((1, 1, 9, 1, 1)))}
            for i in range(2)]}
    first = {"native_best_of_n": {"selected_index": 1, "candidates": [
        {"index": i, "seed": 31 + i, "predicted_value": values[i], "environment_actions": controls[i][0].tolist()}
        for i in range(2)]}, "normalized_actions": controls[1][0].tolist(), "future_slot_rms": 1.0}
    checks["first_regeneration_passes"] = compare_first(proposal(), first, 0)["all_exact"]
    bad = proposal()
    bad["candidates"][0]["actions"] = bad["candidates"][0]["actions"].astype(np.float32)
    checks["first_regeneration_rejects_dtype"] = not compare_first(bad, first, 0)["all_exact"]
    bad = proposal()
    bad["candidates"][0]["value"] = np.nextafter(values[0], np.inf)
    checks["first_regeneration_rejects_one_ulp"] = not compare_first(bad, first, 0)["all_exact"]
    with tempfile.TemporaryDirectory(prefix="cosmos_first_candidate_cpu_") as temporary:
        folder = Path(temporary)
        old, new = folder / "old", folder / "new"
        old.mkdir(); new.mkdir()
        trace = old / "episode.jsonl"
        trace.write_text(json.dumps({"eef_before": [1., 2., 3.]}) + "\n")
        initial, settled = np.zeros(5), np.ones(5)
        np.savez(old / "episode_states.npz", initial=initial, settled=settled)
        (old / "initial.png").write_bytes(b"mock PNG bytes")
        (new / "initial.png").write_bytes(b"mock PNG bytes")
        item = {"episode": {"trace": str(trace), "chunks": str(trace), "initial_image": str(old / "initial.png")}, "first_chunk": first}
        class Env:
            def __init__(self): self.reads = 0
            def get_sim_state(self):
                value = (initial, settled)[min(self.reads, 1)].copy()
                self.reads += 1
                return value
        module = SimpleNamespace(OffScreenRenderEnv=Env)
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return proposal()
        inference = SimpleNamespace(best_of_n=counted)
        obs = {"robot0_eef_pos": np.array([1., 2., 3.])}
        with first_query_override(inference, module, item, 0, new) as tracker:
            env = module.OffScreenRenderEnv()
            env.get_sim_state(); env.get_sim_state()
            one = inference.best_of_n(None, None, None, None, obs)
            two = inference.best_of_n(None, None, None, None, obs)
            checks["first_only_override_and_both_forwards"] = one["selected_index"] == 0 and two["selected_index"] == 1 and len(calls) == 2
            checks["candidate_tensors_unchanged"] = audit.exact_array(one["candidates"][0]["actions"], controls[0])
        checks["normal_exit_restores_hooks"] = inference.best_of_n is counted and module.OffScreenRenderEnv is Env
        try:
            with first_query_override(inference, module, item, 0, new):
                raise RuntimeError("intentional mock exception")
        except RuntimeError:
            pass
        checks["exception_restores_hooks"] = inference.best_of_n is counted and module.OffScreenRenderEnv is Env
    require(all(checks.values()), f"Self-test failure: {checks}")
    return {"status": "passed", "checks": checks, "count": len(checks), "gpu_used": False, "environment_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-registration", type=Path)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Used only when preparing a new exclusive registration")
    parser.add_argument("--report", type=Path, help="Optional new exclusive CPU report file")
    args = parser.parse_args()
    if args.prepare_registration:
        result = prepare(args.prepare_registration.resolve(), args.out_dir)
    elif args.self_test:
        result = self_test()
    elif args.preflight:
        result = preflight(args.config.resolve())[3]
    else:
        require(args.report is None, "--report is a CPU-only option")
        run(args.config.resolve())
        return
    if args.report:
        write(args.report.resolve(), result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
