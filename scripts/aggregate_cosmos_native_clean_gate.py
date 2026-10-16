#!/usr/bin/env python3
"""Audit the registered 10-task x 3-init Cosmos Clean gate without a model.

Live or interrupted runs remain incomplete, even if their current successes
already reach the threshold. Environment traces, native candidate selection,
and all available paired Clean references are independently checked. This
script never changes a runner, protocol, source manifest or episode artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "configs/cosmos_native_decision_gate_v1.json"
GATE_SHA = "338e3be6010593719bcd7dbbaf23ef27b11264dd120fe8f49fdaa5ab4986a2bb"
SUMMARY = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/summary.json"
REFERENCES = (
    ROOT / "runs/cosmos_native_goal_intervention_task0_init012_v1/summary.json",
    ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_v1/summary.json",
)
RESULT_FIELDS = (
    "status", "condition", "suite", "best_of_n", "task_id", "init_id", "task_name", "goal",
    "seed", "environment_seed", "max_steps", "settle_steps", "initial_state_sha256",
    "bddl_sha256", "init_file_sha256", "success_at_start", "success", "terminal_done",
    "environment_steps", "action_chunks", "reward_sum", "final_reward", "action_norm_mean",
)
EXECUTION_FIELDS = {"query_seconds", "cuda_allocated_bytes"}
OPTIONAL_GOAL_FIELDS = {"policy_goal", "forecast_goal", "value_goal"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    # Hash precisely the bytes parsed, even while the producer atomically
    # replaces its summary. Completed episode artifacts are immutable.
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def artifact(path, summary_path):
    p = Path(path)
    for candidate in (p, ROOT / p, summary_path.parent / p, summary_path.parent / p.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def exact_array(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype.str == right.dtype.str and left.shape == right.shape
            and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes())


def array_sha(value):
    array = np.asarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def audit_episode(row, summary_path, gate):
    """Derive success from the stored actual environment trace, not scores."""
    h = gate["clean_health"]
    n = gate["native_best_of_N"]
    expected = {"status": "completed", "condition": "clean", "suite": h["suite"],
                "best_of_n": n["N"], "seed": h["seed"], "environment_seed": h["environment_seed"],
                "max_steps": h["maximum_environment_steps"], "settle_steps": h["settling_steps"],
                "success_at_start": False}
    for key, value in expected.items():
        require(type(row.get(key)) is type(value) and row[key] == value, f"Episode field mismatch: {key}")
    require(integer(row.get("task_id")) and integer(row.get("init_id")), "Task/init identifiers must be integers")
    require(type(row.get("success")) is bool and type(row.get("terminal_done")) is bool,
            "Episode success/done must be Boolean")
    for key in OPTIONAL_GOAL_FIELDS:
        if key in row:
            require(row[key] == row["goal"], f"Clean {key} changed")
    steps = row["environment_steps"]
    require(integer(steps) and 0 < steps <= row["max_steps"], "Invalid environment step count")
    trace_path = artifact(row["trace"], summary_path)
    chunks_path = artifact(row["chunks"], summary_path)
    trace, chunks = records(trace_path), records(chunks_path)
    require(len(trace) == steps, "Trace length disagrees with episode count")
    require([r.get("step") for r in trace] == list(range(1, steps + 1)), "Noncontiguous trace steps")
    require(len(chunks) == row["action_chunks"] == math.ceil(steps / 16), "Chunk count mismatch")
    require([r.get("chunk") for r in chunks] == list(range(1, len(chunks) + 1)), "Chunk order changed")
    for index, chunk in enumerate(chunks):
        first = index * 16 + 1
        require(chunk["first_step"] == first, "Chunk start is not contiguous")
        if "condition" in chunk:
            require(chunk["condition"] == "clean", "Non-clean chunk")
        for key in OPTIONAL_GOAL_FIELDS:
            if key in chunk:
                require(chunk[key] == row["goal"], f"Chunk {key} changed")
        actions = np.asarray(chunk["environment_actions"], dtype=np.float64)
        normalized = np.asarray(chunk["normalized_actions"], dtype=np.float64)
        require(actions.shape == normalized.shape == (16, 7)
                and np.isfinite(actions).all() and np.isfinite(normalized).all(), "Malformed candidate controls")
        require(math.isfinite(chunk["future_slot_rms"]), "Nonfinite forecast RMS")
        selection = chunk["native_best_of_n"]
        candidates = selection["candidates"]
        require(len(candidates) == n["N"], "Candidate set size changed")
        require([c["index"] for c in candidates] == list(range(n["N"])), "Candidate order changed")
        require([c["seed"] for c in candidates] == n["candidate_seeds"], "Candidate seeds changed")
        values = [c["predicted_value"] for c in candidates]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Nonfinite native values")
        selected = selection["selected_index"]
        require(integer(selected) and selected == max(range(n["N"]), key=lambda i: values[i]),
                "Selection is not native argmax with first-candidate tie break")
        for candidate in candidates:
            candidate_actions = np.asarray(candidate["environment_actions"], dtype=np.float64)
            require(candidate_actions.shape == (16, 7) and np.isfinite(candidate_actions).all(),
                    "Malformed candidate action matrix")
        require(exact_array(actions, np.asarray(candidates[selected]["environment_actions"], dtype=np.float64)),
                "Executed chunk differs from selected candidate")
        for offset, event in enumerate(trace[first - 1:first - 1 + 16]):
            require(event["chunk"] == index + 1, "Trace chunk index changed")
            require(exact_array(np.asarray(event["action"], dtype=np.float64), actions[offset]),
                    "Environment action differs from committed selected controls")
    for index, event in enumerate(trace):
        require(type(event.get("task_success")) is bool and type(event.get("done")) is bool,
                "Environment success/done must be Boolean")
        require(math.isfinite(event["reward"]), "Nonfinite environment reward")
        if "condition" in event:
            require(event["condition"] == "clean", "Non-clean environment trace")
        for key, shape in (("action", (7,)), ("eef_before", (3,)), ("eef_after", (3,)),
                           ("eef_quaternion", (4,)), ("gripper_qpos", (2,))):
            value = np.asarray(event[key], dtype=np.float64)
            require(value.shape == shape and np.isfinite(value).all(), f"Malformed physical trace: {key}")
        if index < steps - 1:
            require(not event["task_success"] and not event["done"], "Trace continues past native stopping point")
        if index:
            require(event["eef_before"] == trace[index - 1]["eef_after"], "EEF trace discontinuity")
    require(row["success"] == trace[-1]["task_success"], "Summary success disagrees with environment")
    require(row["terminal_done"] == trace[-1]["done"], "Summary done disagrees with environment")
    require(row["success"] or row["terminal_done"] or steps == row["max_steps"], "Nonterminal episode stopped early")
    rewards = [e["reward"] for e in trace]
    require(row["reward_sum"] == sum(rewards) and row["final_reward"] == rewards[-1], "Reward totals changed")
    action_norm_mean = float(np.mean([np.linalg.norm(np.asarray(e["action"], dtype=np.float64)) for e in trace]))
    require(row["action_norm_mean"] == action_norm_mean, "Recorded action norm differs from actual controls")
    state_path = trace_path.with_name(trace_path.stem + "_states.npz")
    state_hashes = {}
    with np.load(state_path, allow_pickle=False) as states:
        require(set(states.files) == {"initial", "settled", "final", "eef_path"}, "State array catalog changed")
        require(exact_array(states["eef_path"], np.asarray([e["eef_after"] for e in trace], dtype=np.float64)),
                "Archived EEF path disagrees with trace")
        for key in states.files:
            value = states[key]
            require(value.dtype.kind in "fiu" and np.isfinite(value).all(), f"Invalid archived state {key}")
            state_hashes[key] = {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": array_sha(value)}
    paths = {"trace": trace_path, "chunks": chunks_path, "states": state_path,
             **{key: artifact(row[key], summary_path) for key in ("initial_image", "final_image")}}
    return {"status": "valid", "task_id": row["task_id"], "init_id": row["init_id"],
            "success": trace[-1]["task_success"], "environment_steps": steps,
            "steps_to_success": steps if row["success"] else None,
            "termination": "success" if row["success"] else "environment_done" if row["terminal_done"] else "step_limit",
            "artifacts": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
            "state_arrays": state_hashes}


def paired_compare(reference, actual, reference_path, actual_path):
    """Ignore only execution timings and validated clean-routing annotations."""
    checks = {f"episode_{key}": reference.get(key) == actual.get(key) for key in RESULT_FIELDS}
    def strip(rows, chunk=False):
        result = []
        for row in rows:
            allowed = {"condition"} | (EXECUTION_FIELDS | OPTIONAL_GOAL_FIELDS if chunk else set())
            result.append({k: v for k, v in row.items() if k not in allowed})
        return result
    rt, at = artifact(reference["trace"], reference_path), artifact(actual["trace"], actual_path)
    rc, ac = artifact(reference["chunks"], reference_path), artifact(actual["chunks"], actual_path)
    checks["all_trace_fields"] = strip(records(rt)) == strip(records(at))
    checks["all_candidate_values_selections_and_actions"] = strip(records(rc), True) == strip(records(ac), True)
    rp, ap = rt.with_name(rt.stem + "_states.npz"), at.with_name(at.stem + "_states.npz")
    with np.load(rp, allow_pickle=False) as left, np.load(ap, allow_pickle=False) as right:
        checks["state_array_names"] = set(left.files) == set(right.files)
        for key in left.files:
            checks[f"state_{key}_dtype_shape_bytes"] = key in right.files and exact_array(left[key], right[key])
    for key in ("initial_image", "final_image"):
        checks[key + "_png_bytes"] = sha(artifact(reference[key], reference_path)) == sha(artifact(actual[key], actual_path))
    return {"reference_summary": str(reference_path), "task_id": actual["task_id"], "init_id": actual["init_id"],
            "all_exact": all(checks.values()), "checks": checks,
            "reference_trace_sha256": sha(rt), "actual_trace_sha256": sha(at),
            "scope": "Recorded physical trace, both candidates' values/actions, selection, state dtype+shape+bytes and PNG; not full latent-tensor parity"}


def gate_decision(source_status, valid, complete, successes, tasks):
    if source_status != "completed":
        return "incomplete", None
    if not valid or not complete:
        return "invalid", False
    passed = successes >= 24 and tasks >= 8
    return ("go" if passed else "no_go"), passed


def aggregate(summary_path, reference_paths):
    gate, gate_sha = read_json(GATE)
    require(gate_sha == GATE_SHA, "Registered gate changed; refusing to silently change thresholds")
    source, source_sha = read_json(summary_path)
    readiness = gate["broader_clean_readiness"]
    expected = {(t, i) for t in readiness["task_ids"] for i in readiness["init_ids"]}
    require(len(expected) == 30 and readiness["minimum_successes"] == 24
            and readiness["minimum_tasks_with_success"] == 8, "Unexpected registered gate")
    errors = []
    def check(test, message):
        if not test:
            errors.append(message)
    h, n, p = gate["clean_health"], gate["native_best_of_N"], source.get("protocol", {})
    protocol_expected = {"suite": h["suite"], "task_ids": readiness["task_ids"], "init_ids": readiness["init_ids"],
                         "seed": h["seed"], "env_seed": h["environment_seed"], "best_of_n": n["N"],
                         "max_steps": h["maximum_environment_steps"], "settle_steps": h["settling_steps"]}
    for key, value in protocol_expected.items():
        check(type(p.get(key)) is type(value) and p.get(key) == value, f"Run protocol mismatch: {key}")
    check(source.get("condition") == "clean", "Run is not Clean")
    provenance = source.get("provenance", {})
    check(provenance.get("checkpoint_sha256") == gate["checkpoint_sha256"], "Checkpoint provenance mismatch")
    for key, value in {"sampler_seeds": n["candidate_seeds"], "action_steps": n["action_denoising_steps"],
                       "future_steps": n["future_denoising_steps"], "value_steps": n["value_denoising_steps"],
                       "training_updates": 0, "api_requests": 0}.items():
        check(provenance.get(key) == value, f"Run provenance mismatch: {key}")
    planned = [(v["task_id"], v["init_id"]) for v in source.get("planned_episodes", [])]
    check(len(planned) == 30 and set(planned) == expected, "Planned episodes differ from registered 10x3 grid")
    rows, audits = {}, []
    for row in source.get("results", []):
        key = row.get("task_id"), row.get("init_id")
        if key not in expected or key in rows:
            errors.append(f"Unexpected or duplicate episode: {key}")
            continue
        rows[key] = row
        try:
            audits.append(audit_episode(row, summary_path, gate))
        except Exception as exc:
            audits.append({"status": "invalid", "task_id": key[0], "init_id": key[1],
                           "reported_success": row.get("success"), "error_type": type(exc).__name__, "error": str(exc)})
            errors.append(f"Episode {key} audit failed: {exc}")
    references, pairs = [], []
    expected_pair_keys, paired_keys = set(), set()
    for reference_path in reference_paths:
        ref, ref_sha = read_json(reference_path)
        require(ref.get("status") == "completed", "Paired reference must be completed")
        references.append({"path": str(reference_path), "sha256": ref_sha})
        reference_rows = [r for r in ref.get("results", []) if r.get("condition") == "clean" and r.get("task_id") == 0]
        require(reference_rows, "No Task0 Clean references")
        seen_reference = set()
        for row in reference_rows:
            key = row["task_id"], row["init_id"]
            require(key in expected and key not in seen_reference, "Unexpected/duplicate Clean reference")
            seen_reference.add(key)
            pair_key = str(reference_path), key
            expected_pair_keys.add(pair_key)
            if key not in rows:
                pairs.append({"reference_summary": str(reference_path), "task_id": key[0], "init_id": key[1],
                              "status": "pending", "all_exact": None})
                continue
            try:
                audit_episode(row, reference_path, gate)
                pair = paired_compare(row, rows[key], reference_path, summary_path)
                pair["status"] = "exact" if pair["all_exact"] else "different"
                pairs.append(pair)
                if pair["all_exact"]:
                    paired_keys.add(pair_key)
            except Exception as exc:
                pairs.append({"reference_summary": str(reference_path), "task_id": key[0], "init_id": key[1],
                              "status": "invalid", "all_exact": False, "error_type": type(exc).__name__, "error": str(exc)})
    validated = [r for r in audits if r["status"] == "valid"]
    successes = sum(r["success"] for r in validated)
    success_tasks = len({r["task_id"] for r in validated if r["success"]})
    complete = set(rows) == expected and len(audits) == 30 and len(validated) == 30
    pairing = expected_pair_keys == paired_keys
    decision, passed = gate_decision(source.get("status"), not errors, complete, successes, success_tasks)
    per_task = []
    for task in readiness["task_ids"]:
        task_rows = [r for r in validated if r["task_id"] == task]
        completed = len(task_rows)
        per_task.append({"task_id": task, "planned_episodes": 3, "validated_completed": completed,
                         "successes": sum(r["success"] for r in task_rows),
                         "success_rate_completed": sum(r["success"] for r in task_rows) / completed if completed else None,
                         "success_rate_registered_three": sum(r["success"] for r in task_rows) / 3 if completed == 3 else None,
                         "mean_environment_steps_completed": float(np.mean([r["environment_steps"] for r in task_rows])) if completed else None,
                         "mean_steps_to_success": float(np.mean([r["environment_steps"] for r in task_rows if r["success"]])) if any(r["success"] for r in task_rows) else None,
                         "episodes": [{k: r[k] for k in ("init_id", "success", "environment_steps", "steps_to_success", "termination")} for r in task_rows]})
    return {"schema_version": 1, "status": decision, "gate_passed": passed,
            "source_status": source.get("status"), "source_summary": str(summary_path), "source_sha256": source_sha,
            "registered_gate": str(GATE), "registered_gate_sha256": gate_sha,
            "analysis_script_sha256": sha(__file__), "numpy_version": np.__version__,
            "threshold": {"minimum_successes": 24, "registered_episodes": 30, "minimum_tasks_with_success": 8},
            "validated_completed": len(validated), "successes": successes,
            "success_rate_completed": successes / len(validated) if validated else None,
            "success_rate_registered_30": successes / 30 if complete else None,
            "tasks_with_success": success_tasks, "complete_registered_grid": complete,
            "missing_episodes": [{"task_id": t, "init_id": i} for t, i in sorted(expected - set(rows))],
            "validation_errors": errors, "episodes": audits, "per_task": per_task,
            "paired_reference_sources": references, "paired_references": pairs,
            "all_available_paired_references_exact": pairing,
            "paired_reference_scope": "Independent reproducibility audit; not an added condition or a changed threshold in the registered Clean readiness gate.",
            "failure_policy": "Every registered completed failure is retained; missing or runtime-error episodes are not scored as success/failure.",
            "decision_scope": "Development Clean readiness only; not held-out population performance, Goal-invariance or attack evidence.",
            "incomplete_policy": "Any source status other than completed has status=incomplete and gate_passed=null, regardless of observed success count.",
            "api_requests": 0, "training_updates": 0, "gpu_used": False}


def self_test():
    checks = {
        "running_cannot_pass": gate_decision("running", True, True, 30, 10) == ("incomplete", None),
        "error_cannot_pass": gate_decision("error", True, False, 24, 8) == ("incomplete", None),
        "exact_boundary_passes": gate_decision("completed", True, True, 24, 8) == ("go", True),
        "below_success_threshold_fails": gate_decision("completed", True, True, 23, 10) == ("no_go", False),
        "below_task_threshold_fails": gate_decision("completed", True, True, 24, 7) == ("no_go", False),
        "missing_episode_fails": gate_decision("completed", True, False, 24, 8) == ("invalid", False),
        "corrupt_trace_fails": gate_decision("completed", False, True, 30, 10) == ("invalid", False),
        "dtype_matters": not exact_array(np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)),
        "signed_zero_matters": not exact_array(np.array([0.]), np.array([-0.])),
        "shape_matters": not exact_array(np.array([[1.]]), np.array([1.])),
        "same_bytes_pass": exact_array(np.array([1.]), np.array([1.])),
    }
    require(all(checks.values()), f"Self-test failed: {checks}")
    return {"status": "passed", "checks": checks, "count": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--reference", type=Path, action="append", help="Repeat for independent paired reference summaries")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/cosmos_native_clean_gate_audit_v2/analysis.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    references = [p.resolve() for p in args.reference] if args.reference else [p for p in REFERENCES if p.exists()]
    require(references, "At least one completed paired Clean reference is required")
    result = aggregate(args.summary.resolve(), references)
    write_json(args.out.resolve(), result)
    print(json.dumps({k: result[k] for k in ("status", "gate_passed", "source_status", "validated_completed", "successes", "tasks_with_success", "all_available_paired_references_exact", "validation_errors")}, indent=2))
    print(str(args.out.resolve()))


if __name__ == "__main__":
    main()
