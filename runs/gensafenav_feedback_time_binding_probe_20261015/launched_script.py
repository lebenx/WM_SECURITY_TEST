#!/usr/bin/env python3
"""Synthetic CPU probe of the original GenSafeNav forecast/label index operator.

Extract Human.update_aci unchanged from its cached AST. No Human constructor,
environment, trained predictor, learned policy, or ACI adaptation is executed.
ACI sinks record the residual inputs produced by that original method.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gensafenav_feedback_time_binding_probe_20261015.json"
SOURCES = {
    "human": "crowd_sim/envs/utils/human.py",
    "step": "crowd_sim/envs/crowd_sim_pred.py",
    "prediction_receiver": "crowd_sim/envs/crowd_sim_pred_real_gst.py",
    "human_reset": "crowd_sim/envs/crowd_sim_var_num.py",
    "wrapper": "rl/vec_env/vec_pretext_normalize.py",
    "trained_config": "trained_models/Ours_GST/configs/config.py",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def get_method(source):
    tree = ast.parse(source)
    human = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Human")
    method = next(node for node in human.body if isinstance(node, ast.FunctionDef) and node.name == "update_aci")
    require(not method.decorator_list, "Unexpected decorators on source method")
    return method


class ForecastTable:
    def __init__(self, origin, values, context):
        self.origin, self.values, self.context = origin, values, context

    def __getitem__(self, horizon):
        require(type(horizon) is int and 0 <= horizon <= 5, "Unexpected forecast horizon access")
        require(self.context.get("pending_access") is None, "Unconsumed prior forecast access")
        self.context["pending_access"] = {
            "issue_index": self.origin,
            "selected_history_index": self.context["selected_history_index"],
            "forecast_horizon": horizon,
            "forecast_target_index": self.origin + horizon,
            "forecast_position": self.values[horizon].tolist(),
        }
        return self.values[horizon]


class ForecastHistory(list):
    def __init__(self, context):
        super().__init__()
        self.context = context

    def __getitem__(self, index):
        require(type(index) is int and index < 0, "Unexpected source history lookup")
        self.context["selected_history_index"] = index
        return super().__getitem__(index)


class ResidualSink:
    def __init__(self, horizon, context, records, dt):
        self.horizon, self.context, self.records, self.dt = horizon, context, records, dt

    def update_true_value(self, value):
        accessed = self.context.pop("pending_access")
        require(accessed is not None and accessed["forecast_horizon"] == self.horizon,
                "Source forecast access and ACI sink horizon differ")
        label_index = self.context["label_index"]
        self.records.append({
            **accessed, "aci_horizon": self.horizon,
            "label_index": label_index, "label_position": self.context["label_position"].tolist(),
            "issue_seconds": accessed["issue_index"] * self.dt,
            "forecast_target_seconds": accessed["forecast_target_index"] * self.dt,
            "label_seconds": label_index * self.dt,
            "target_time_matches_label": accessed["forecast_target_index"] == label_index,
            "actual_native_residual": float(value),
        })


def run_condition(method, valid_schedule, forecast_values, dt, activity):
    context, native_records, reference_records, skipped = {}, [], [], []
    state = SimpleNamespace(pred_horizon_aci=5, gt_locations_aci=[],
                            predictions_aci=ForecastHistory(context))
    state.pred_error_aci_list = [ResidualSink(h, context, native_records, dt) for h in range(1, 6)]
    issued, observed_labels, events = {}, [], []
    for origin, valid in enumerate(valid_schedule):
        event = {"issue_index": origin, "prediction_valid": valid,
                 "native_history_length_before": len(state.predictions_aci)}
        if valid:
            # Matches talk2Env's valid append, then CrowdSimPred.step's next-position label.
            values = forecast_values[origin].copy()
            state.predictions_aci.append(ForecastTable(origin, values, context))
            issued[origin] = values
            label_index = origin + 1
            position = np.asarray([dt * label_index, 0.0], dtype=np.float64)
            state.gt_locations_aci.append(position)
            observed_labels.append({"index": label_index, "position": position.tolist()})
            context.update(label_index=label_index, label_position=position, pending_access=None)
            before = len(native_records)
            require(activity["native_update_aci_calls"] < 6, "Frozen operator budget exceeded")
            activity["native_update_aci_calls"] += 1
            method(state)
            require(context.get("pending_access") is None, "Source left an unmatched forecast access")
            event.update(label_index=label_index, native_updates=len(native_records) - before)
            # Independent timestamp join: same actually issued forecasts and same observed label.
            for horizon in range(1, 6):
                required_origin = label_index - horizon
                if required_origin not in issued:
                    skipped.append({"label_index": label_index, "horizon": horizon,
                                    "required_issue_index": required_origin,
                                    "reason": "No issued forecast for this origin; no relabeling or invented label."})
                    continue
                forecast_position = issued[required_origin][horizon]
                residual = float(np.sqrt(sum(float(a - b) ** 2 for a, b in zip(position, forecast_position))))
                reference_records.append({"label_index": label_index, "horizon": horizon,
                                          "issue_index": required_origin,
                                          "forecast_target_index": required_origin + horizon,
                                          "label_position": position.tolist(),
                                          "forecast_position": forecast_position.tolist(), "residual": residual})
        else:
            event.update(label_index=None, native_updates=0,
                         invalid_semantics="No prediction/GT append and no native update; prior history retained.")
        event["native_history_length_after"] = len(state.predictions_aci)
        events.append(event)
    native_pairs = {(r["label_index"], r["aci_horizon"]) for r in native_records}
    reference_pairs = {(r["label_index"], r["horizon"]) for r in reference_records}
    return {
        "valid_schedule": valid_schedule, "issued_indices": list(issued), "observed_labels": observed_labels,
        "events": events, "native_updates": native_records, "timestamp_reference_updates": reference_records,
        "timestamp_reference_skips": skipped,
        "native_time_mismatches": [r for r in native_records if not r["target_time_matches_label"]],
        "native_only_label_horizon_pairs": sorted(map(list, native_pairs - reference_pairs)),
        "reference_only_label_horizon_pairs": sorted(map(list, reference_pairs - native_pairs)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    require(config["status"] == "frozen_before_operator_execution"
            and config["query_times"] == [0, 1, 2, 3] and config["time_step_seconds"] == 0.25
            and config["conditions"] == {"continuous_valid": [True] * 4,
                                         "two_step_validity_gap": [True, False, False, True]}
            and config["max_native_update_aci_calls"] == 6
            and config["max_models_or_environment_or_api_calls"] == 0, "Frozen operator panel changed")
    source_root = ROOT / config["source_root"]
    metadata = json.loads((source_root.parent / "resource_source_metadata.json").read_text())
    require(metadata["revision"] == config["source_commit"], "Source revision differs")
    source_text = {key: (source_root / path).read_text() for key, path in SOURCES.items()}
    # Fail closed if the exact reviewed valid/invalid ownership blocks no longer exist.
    require("if human.prediction_valid:\n                human.predictions_aci.append(deepcopy(predictions))\n                human.last_prediction = deepcopy(predictions)\n            else: \n                human.last_prediction = None"
            in source_text["prediction_receiver"], "Reviewed no-reset invalid receiver branch changed")
    require("if h.prediction_valid:\n                h.gt_locations_aci.append(np.array([h.px, h.py]))\n                h.update_aci()"
            in source_text["step"], "Reviewed valid-label caller changed")
    node = get_method(source_text["human"])
    output = (args.out or ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "launched_script.py")
    shutil.copy2(args.config, output / "launched_config.json")
    for path in SOURCES.values():
        target = output / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / path, target)
    started = time.monotonic()
    activity = {"native_update_aci_calls": 0, "native_residual_sink_calls": 0,
                "aci_adaptation_calls": 0, "model_calls": 0, "environment_calls": 0,
                "api_calls": 0, "gpu_calls": 0, "checkpoint_loads": 0}
    report = {
        "status": "running", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve().relative_to(ROOT)), "source_commit": config["source_commit"],
        "scope": config["scope"], "activity": activity, "conditions": {},
        "source_method": {"path": str((source_root / SOURCES["human"]).relative_to(ROOT)),
                          "first_line": node.lineno, "last_line": node.end_lineno,
                          "extraction": "Original FunctionDef copied unchanged into an otherwise empty AST module.",
                          "text": ast.get_source_segment(source_text["human"], node)},
        "caller_source_anchors": [f"{config['source_root']}/{SOURCES[k]}:{line}" for k, line in
                                  (("prediction_receiver", 94), ("step", 220), ("human_reset", 329), ("wrapper", 60))],
        "caller_scope": "Reviewed source retains history across invalid predictions. Synthetic fixture stipulates prediction_valid; no natural occurrence, visibility mapping, or full caller execution is claimed.",
        "limits": ["No actual trained trajectory predictor, ACI quantile adaptation, actor, physical environment, collision, or utility measurement.",
                   "ACI sinks only record the residuals emitted by the original method; they do not simulate learned calibration.",
                   "out_mask=false is not assumed equivalent to human.prediction_valid=false.",
                   "Timestamp joining is an ordinary attribution correction, not evidence of a new security mechanism or effective defense."],
    }
    code = 1
    try:
        namespace = {"np": np, "__builtins__": {"len": len, "min": min, "range": range}}
        module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
        exec(compile(module, str(source_root / SOURCES["human"]), "exec"), namespace)
        method = namespace["update_aci"]
        dt = config["time_step_seconds"]
        forecast_values = {origin: np.asarray([[dt * (origin + horizon), 0.0] for horizon in range(6)],
                                              dtype=np.float64) for origin in config["query_times"]}
        report["synthetic_forecast_tables"] = {str(k): v.tolist() for k, v in forecast_values.items()}
        for name, schedule in config["conditions"].items():
            result = run_condition(method, schedule, forecast_values, dt, activity)
            report["conditions"][name] = result
            activity["native_residual_sink_calls"] += len(result["native_updates"])
            if name == "continuous_valid":
                report["continuous_control_passed"] = (
                    not result["native_time_mismatches"] and not result["native_only_label_horizon_pairs"]
                    and not result["reference_only_label_horizon_pairs"]
                    and all(r["actual_native_residual"] == 0.0 for r in result["native_updates"])
                    and all(r["residual"] == 0.0 for r in result["timestamp_reference_updates"]))
                require(report["continuous_control_passed"], "Continuous control failed; stop before gap attribution")
        require(activity["native_update_aci_calls"] == 6, "Frozen panel incomplete")
        gap = report["conditions"]["two_step_validity_gap"]
        report["result"] = {"gap_time_mismatch_observed": bool(gap["native_time_mismatches"]),
                            "mismatched_native_updates": len(gap["native_time_mismatches"]),
                            "timestamp_reference_residuals_all_zero": all(r["residual"] == 0.0 for r in gap["timestamp_reference_updates"]),
                            "interpretation": "Operator-level forecast-label time binding only; no uncertainty, task harm, or novelty inference."}
        report["status"] = "completed_synthetic_operator_probe"
        code = 0
    except BaseException as error:
        report["status"] = "invalid_operator_probe"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = time.monotonic() - started
    report["exit_code"] = code
    write_json(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "result": report.get("result"),
                      "activity": activity, "output": str(output.relative_to(ROOT))}, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
