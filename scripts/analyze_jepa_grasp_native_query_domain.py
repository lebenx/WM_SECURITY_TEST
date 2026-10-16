#!/usr/bin/env python3
"""Analyze saved native GRASP queries; never load a model or run a forward pass.

Run with .venv-jepa-native/bin/python when query1.pt is available. Only the
first seed/query and the four recorded loss-entry stages are inspected. Patch
counts are descriptive, dependent observations, not samples or attack rates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGES = (0, 1, 49, 50)
TOLERANCE = 1e-5
HORIZON = 6
TERMINAL_STATUSES = {"completed_clean_qualification", "completed_futility_stop", "resource_deferred", "error"}
CSV_FIELDS = (
    "family", "stage", "scope", "step", "input_role", "metric", "count_unit",
    "consumer_execution_status", "patch_count",
    "max_abs", "mean_abs", "rms", "p50_abs", "p95_abs",
    "above_tolerance_count", "fraction_above_tolerance", "tolerance",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path, payload):
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def finite_array(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in "fiu" and np.isfinite(array).all(),
            f"{name}: expected finite numeric data")
    return array


def load_npz(path, report):
    if not path.is_file():
        report["missing_inputs"].append(str(path))
        return None
    try:
        payload = path.read_bytes()
        report["input_files"].append(binding(path, payload))
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except Exception as exc:
        report["input_errors"].append({"path": str(path), "error": str(exc)})
        return None


def prepare_constraints(data):
    shapes = {"encoder_normal": (384,), "encoder_offset": (384,),
              "predictor_normal": (400,), "predictor_offset": (400,)}
    result = {}
    for name, shape in shapes.items():
        array = finite_array(data[name], name).astype(np.float64)
        require(array.shape == shape, f"{name}: expected shape {shape}, got {array.shape}")
        result[name] = array
    for name in ("encoder_normal", "predictor_normal"):
        require(np.isclose(np.linalg.norm(result[name]), 1.0, rtol=1e-8, atol=1e-10),
                f"{name}: saved normal is not unit length")
    return result


def prepare_context(data):
    shape_array = finite_array(data["visual_shape"], "visual_shape")
    require(shape_array.ndim == 1 and shape_array.size == 4
            and np.equal(shape_array, np.floor(shape_array)).all(),
            "visual_shape: expected four integer dimensions")
    shape = tuple(int(x) for x in shape_array)
    require(all(x > 0 for x in shape) and shape[-1] == 384,
            "visual_shape: expected positive V,H,W,384 dimensions")
    s0 = finite_array(data["s_0_flat"], "s_0_flat")
    goal = finite_array(data["g_flat"], "g_flat")
    proprio = finite_array(data["initial_proprio"], "initial_proprio")
    require(s0.ndim == 2 and s0.shape[0] == 1 and s0.shape[1] == int(np.prod(shape)),
            "s_0_flat: expected first native query batch=1 and matching visual shape")
    require(goal.shape == s0.shape, "g_flat: shape differs from initial visual state")
    patches = int(np.prod(shape[:-1]))
    require(proprio.ndim == 4 and proprio.shape[0] == 1 and proprio.shape[1] >= 1
            and proprio.shape[2:] == (patches, 16),
            "initial_proprio: expected B,context,matching_patch,16")
    return {"s0": s0, "goal": goal, "proprio": proprio[:, -1],
            "visual_shape": shape, "patches": patches, "batch": 1}


def residuals(visual, proprio, constraints):
    """Accept aligned B,time,patch,feature arrays; never average before testing."""
    require(visual.ndim == proprio.ndim == 4 and visual.shape[:-1] == proprio.shape[:-1]
            and visual.shape[-1] == 384 and proprio.shape[-1] == 16,
            "Visual/proprio patch alignment differs")
    visual = finite_array(visual, "query visual").astype(np.float64)
    proprio = finite_array(proprio, "query proprio").astype(np.float64)
    encoder = (visual - constraints["encoder_offset"]) @ constraints["encoder_normal"]
    joint = np.concatenate((visual, proprio), axis=-1)
    predictor = (joint - constraints["predictor_offset"]) @ constraints["predictor_normal"]
    union = np.minimum(np.abs(encoder), np.abs(predictor))
    return {"encoder": encoder, "predictor": predictor, "union": union}


def statistics(values):
    absolute = np.abs(values).reshape(-1)
    require(absolute.size > 0 and np.isfinite(absolute).all(), "Empty/nonfinite residuals")
    count = int(absolute.size)
    above = int(np.count_nonzero(absolute > TOLERANCE))
    return {"patch_count": count, "max_abs": float(absolute.max()),
            "mean_abs": float(absolute.mean()), "rms": float(np.sqrt(np.mean(absolute ** 2))),
            "p50_abs": float(np.quantile(absolute, 0.5)),
            "p95_abs": float(np.quantile(absolute, 0.95)),
            "above_tolerance_count": above, "fraction_above_tolerance": above / count,
            "tolerance": TOLERANCE}


def summarize(values, family, stage, rows, virtual):
    result = {"all_inputs": {}, "successor_or_virtual_inputs": {}, "steps": []}
    for metric, array in values.items():
        for scope, selected in (("all_inputs", array), ("successor_or_virtual_inputs", array[:, 1:])):
            stats = statistics(selected)
            result[scope][metric] = stats
            rows.append({"family": family, "stage": stage, "scope": scope,
                         "step": "", "input_role": "", "metric": metric, **stats})
    for step in range(values["union"].shape[1]):
        role = "encoded_root" if step == 0 else ("virtual_state" if virtual else "paired_successor")
        entry = {"step": step, "input_role": role, "metrics": {}}
        for metric, array in values.items():
            stats = statistics(array[:, step])
            entry["metrics"][metric] = stats
            rows.append({"family": family, "stage": stage, "scope": "per_step",
                         "step": step, "input_role": role, "metric": metric, **stats})
        result["steps"].append(entry)
    return result


def analyze_forecast(path, context, constraints, report, rows):
    """Deserialize only the runner's local tensor artifact, with CPU mapping."""
    if not path.is_file():
        report["missing_inputs"].append(str(path))
        return {"status": "missing", "path": str(path)}
    try:
        import torch

        payload = path.read_bytes()
        report["input_files"].append(binding(path, payload))
        artifact = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        report["activity"]["local_tensor_artifact_deserializations"] += 1
        expected_keys = {"normalized_macro_actions", "physical_actions", "losses",
                         "forecast", "rng_before", "rng_after"}
        require(isinstance(artifact, dict) and set(artifact) == expected_keys,
                "query1.pt does not match the runner's tensor-artifact schema")

        def tensor_tree(value):
            if isinstance(value, torch.Tensor):
                require(value.device.type == "cpu", "Artifact tensor did not map to CPU")
            else:
                require(isinstance(value, dict) and all(isinstance(key, str) for key in value),
                        "Artifact contains data outside a dictionary/tensor tree")
                for child in value.values():
                    tensor_tree(child)

        tensor_tree(artifact)
        forecast = artifact["forecast"]
        require(set(forecast) == {"visual", "proprio"}, "Unexpected forecast fields")
        visual = forecast["visual"].detach().numpy()
        proprio = forecast["proprio"].detach().numpy()
        require(visual.shape == (HORIZON + 1, context["batch"], *context["visual_shape"])
                and proprio.shape == (HORIZON + 1, context["batch"], context["patches"], 16),
                "Final forecast must contain the initial prefix and six paired successors")
        visual = visual.swapaxes(0, 1).reshape(context["batch"], HORIZON + 1, context["patches"], 384)
        proprio = proprio.swapaxes(0, 1)
        same_visual = np.array_equal(visual[:, 0].reshape(context["s0"].shape), context["s0"])
        same_proprio = np.array_equal(proprio[:, 0], context["proprio"])
        values = residuals(visual, proprio, constraints)
        result = {"status": "completed", "path": str(path),
                  "context_binding": {"initial_visual_exact_match": same_visual,
                                      "initial_proprio_exact_match": same_proprio},
                  "stats": summarize(values, "paired_full_rollout", "query1_final", rows, False),
                  "protected_domain_checks": {
                      "initial_prefix_union_within_tolerance": bool(np.all(values["union"][:, 0] <= TOLERANCE)),
                      "successors_predictor_within_tolerance": bool(np.all(np.abs(values["predictor"][:, 1:]) <= TOLERANCE)),
                  }}
        if not (same_visual and same_proprio):
            result["status"] = "invalid_context_binding"
            report["input_errors"].append({"path": str(path), "error": "Final forecast prefix differs from recorded context"})
        return result
    except Exception as exc:
        report["input_errors"].append({"path": str(path), "error": str(exc)})
        return {"status": "unavailable_or_invalid", "path": str(path), "error": str(exc)}


def completed_loss_evidence(summary):
    """The runner emits a heartbeat only after original_grasp_loss returns."""
    for episode_index, episode in enumerate(summary.get("episodes", [])):
        if episode.get("seed") != 1:
            continue
        for query_index, query in enumerate(episode.get("queries", [])):
            if query.get("index") != 1:
                continue
            heartbeat = query.get("latest_heartbeat", {})
            completed = heartbeat.get("completed_grasp_loss_evaluations")
            if (heartbeat.get("seed") == 1 and heartbeat.get("query") == 1
                    and type(completed) is int and 0 <= completed <= 500):
                return {"status": "available", "completed_loss_evaluations": completed,
                        "source": "summary.json",
                        "json_path": f"episodes[{episode_index}].queries[{query_index}].latest_heartbeat",
                        "rule": "For zero-based stage k, completed_loss_evaluations > k confirms all six one-step calls returned.",
                        "writer": "scripts/pilot_jepa_grasp_native_clean.py:237-246; heartbeat follows original_grasp_loss return"}
    return {"status": "unavailable", "completed_loss_evaluations": None}


def analyze(run_dir, constraints_path, run_summary):
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial", "run_directory": str(run_dir), "seed": 1, "query": 1,
        "fixed_loss_entry_stages": list(STAGES), "tolerance": TOLERANCE,
        "arithmetic": "Saved values converted to float64; saved unit-normal affine constraints used unchanged.",
        "scope": "Offline structural-domain audit of saved native queries, not a model test, attack, utility evaluation or invariance proof.",
        "counting": "Counts and fractions describe dependent patch occurrences within one root/query. Neither patches, time steps nor stages are independent samples; fractions are not attack rates.",
        "construction": "For each stage, concatenate encoded s0 and five recorded virtual visual states. Pair every input with the last initial-proprio frame, broadcast across six steps exactly as native _make_context does. Goal is a loss target, not a seventh query.",
        "recording_boundary": "Snapshots were saved at loss entry, before its six world-model calls. A later heartbeat is written only after original_grasp_loss returns: completed_loss_evaluations > k confirms stage k was fully consumed. A complete, bound query1.pt can also confirm consumption. Otherwise the snapshot alone does not prove all calls completed.",
        "formula": {"encoder": "(visual-encoder_offset) dot encoder_normal",
                    "predictor": "(concat(visual,proprio)-predictor_offset) dot predictor_normal",
                    "union": "min(abs(encoder),abs(predictor)) per patch, before any aggregation"},
        "input_files": [], "missing_inputs": [], "input_errors": [], "stages": [],
        "activity": {"local_tensor_artifact_deserializations": 0, "model_checkpoint_loads": 0,
                     "model_forward_calls": 0, "environment_calls": 0, "model_api_calls": 0,
                     "network_requests": 0},
        "source_notes": str(ROOT / "reports/grasp_rollout_equivalence_mechanism_20261016.json"),
    }
    rows = []
    episode = run_dir / "seed1_clean"
    constraints_data = load_npz(constraints_path, report)
    context_path = episode / "native_virtual_context.npz"
    context_data = load_npz(context_path, report)
    constraints = context = None
    for data, path, prepare, label in ((constraints_data, constraints_path, prepare_constraints, "constraints"),
                                       (context_data, context_path, prepare_context, "context")):
        if data is not None:
            try:
                prepared = prepare(data)
                if label == "constraints":
                    constraints = prepared
                else:
                    context = prepared
                    report["context_shapes"] = {"s0": list(context["s0"].shape),
                                                "visual": list(context["visual_shape"]),
                                                "initial_proprio_last_frame": list(context["proprio"].shape)}
            except Exception as exc:
                report["input_errors"].append({"path": str(path), "error": str(exc)})
    for stage in STAGES:
        path = episode / f"native_virtual_input_k{stage}.npz"
        data = load_npz(path, report)
        entry = {"stage": stage, "path": str(path),
                 "status": ("unavailable_or_invalid" if path.is_file() else "missing")
                 if data is None else "blocked_by_context_or_constraints"}
        if data is not None and context is not None and constraints is not None:
            try:
                states = finite_array(data["s_flat"], "s_flat")
                actions = finite_array(data["actions"], "actions")
                require(states.shape == (context["batch"], HORIZON - 1, context["s0"].shape[1])
                        and actions.shape == (context["batch"], HORIZON, 20),
                        "Expected five virtual states and six 20-dimensional action blocks")
                visual = np.concatenate((context["s0"][:, None], states), axis=1)
                visual = visual.reshape(context["batch"], HORIZON, context["patches"], 384)
                proprio = np.broadcast_to(context["proprio"][:, None],
                                          (context["batch"], HORIZON, context["patches"], 16))
                entry.update(status="completed", action_shape=list(actions.shape),
                             stats=summarize(residuals(visual, proprio, constraints),
                                             "recorded_loss_entry_input", stage, rows, True))
            except Exception as exc:
                entry.update(status="invalid", error=str(exc))
                report["input_errors"].append({"path": str(path), "error": str(exc)})
        report["stages"].append(entry)
    forecast_path = episode / "query1.pt"
    if context is not None and constraints is not None:
        report["final_paired_rollout"] = analyze_forecast(forecast_path, context, constraints, report, rows)
    else:
        report["final_paired_rollout"] = {"status": "blocked_by_context_or_constraints", "path": str(forecast_path)}
        if not forecast_path.is_file():
            report["missing_inputs"].append(str(forecast_path))
    report["virtual_panel_complete"] = all(stage["status"] == "completed" for stage in report["stages"])
    evidence = completed_loss_evidence(run_summary)
    report["native_consumption_evidence"] = evidence
    completed = evidence["completed_loss_evaluations"]
    for entry in report["stages"]:
        if entry["status"] != "completed":
            consumption = "snapshot_unavailable_or_invalid"
        elif completed is not None and completed > entry["stage"]:
            consumption = "all_six_native_one_step_calls_confirmed_by_completed_loss_heartbeat"
        elif report["final_paired_rollout"]["status"] == "completed":
            consumption = "complete_first_query_artifact_verified"
        else:
            consumption = "loss_entry_snapshot_only_consumption_not_confirmed"
        entry["consumer_execution_status"] = consumption
    stage_status = {entry["stage"]: entry["consumer_execution_status"] for entry in report["stages"]}
    for row in rows:
        row["consumer_execution_status"] = (stage_status[row["stage"]]
                                             if row["family"] == "recorded_loss_entry_input"
                                             else "saved_final_paired_rollout")
    if report["virtual_panel_complete"] and report["final_paired_rollout"]["status"] == "completed":
        report["status"] = "completed_saved_query_domain_analysis"
    report["csv_row_count"] = len(rows)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/jepa_grasp_native_clean_20261016")
    parser.add_argument("--constraints", type=Path, default=ROOT / "runs/jepa_affine_query_boundary_20261016/constraints.npz")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    try:
        summary_payload = summary_path.read_bytes()
        summary = json.loads(summary_payload)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "partial_run_terminal_unconfirmed", "error": str(exc),
                          "outputs_written": False, "reason": "Do not create or occupy an active run's analysis outputs"}))
        return
    if summary.get("status") not in TERMINAL_STATUSES or summary.get("active") is not None:
        print(json.dumps({"status": "deferred_run_not_terminal", "run_status": summary.get("status"),
                          "active": summary.get("active"), "outputs_written": False}))
        return
    output_dir = run_dir / "analysis"
    json_path = output_dir / "native_query_domain.json"
    csv_path = output_dir / "native_query_domain.csv"
    require(not json_path.exists() and not csv_path.exists(), "Analysis output already exists; refusing to overwrite")
    report, rows = analyze(run_dir, args.constraints.resolve(), summary)
    report["run_summary"] = {**binding(summary_path, summary_payload), "status": summary["status"],
                             "active_at_stop": summary.get("active_at_stop"),
                             "scope": "Terminal status only; this script does not independently audit task utility or the Clean gate."}
    report["script"] = binding(Path(__file__).resolve(), Path(__file__).read_bytes())
    report["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
    output_dir.mkdir(exist_ok=True)
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({**row, "count_unit": "dependent_patch_occurrence"} for row in rows)
    with json_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "outputs": report["outputs"],
                      "missing_inputs": report["missing_inputs"], "input_errors": report["input_errors"]}))


if __name__ == "__main__":
    main()
