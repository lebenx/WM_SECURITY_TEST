#!/usr/bin/env python3
"""Score four cached forecasts with the native objective; no model or environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports/dino_first_query_two_mean_protocol_20261015.json"
OUTPUT = ROOT / "reports/dino_first_query_two_mean_ranking_20261015.json"
CANDIDATES = ("clean_identity_copy", "canonical_request")
REPRESENTATIONS = ("original", "canonical")
FEATURES = ("visual", "proprio")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path, np):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(out):
    require(not out.exists(), "Refusing to overwrite an existing result")
    started = time.monotonic()
    protocol = read(PROTOCOL)
    report = {
        "status": "checking_sources", "protocol": str(PROTOCOL.relative_to(ROOT)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_order": list(CANDIDATES), "representation_order": list(REPRESENTATIONS),
        "scope": protocol["selection"]["candidate_definition"],
        "interpretation_limits": protocol["interpretation_limits"],
        "activity": {**protocol["activity_limits"], "native_objective_calls": 0},
        "checks": {}, "cost_matrix": None, "rankings": None,
        "candidate_actual_utilities": None, "selected_utility_contrast": None,
    }
    failed = False
    try:
        import numpy as np
        import torch
        from tensordict import TensorDict
        import yaml
        torch.set_num_threads(4)
        require(protocol["status"] == "frozen_before_new_objective_cost_computation", "Protocol not frozen")
        require(protocol["selection"]["query"] == 1 and protocol["selection"]["candidate_order"] == list(CANDIDATES), "Candidate bank changed")
        require(protocol["selection"]["representations"] == list(REPRESENTATIONS), "Representations changed")
        require(protocol["scoring"]["alpha"] == .1 and protocol["scoring"]["sum_all_diffs"] is False
                and protocol["scoring"]["calls"] == 4, "Frozen objective changed")
        native = ROOT / protocol["selection"]["native_run"]
        diagnostic = ROOT / protocol["selection"]["forecast_run"]
        source = read(native / "summary.json")
        calibration = read(diagnostic / "summary.json")
        require(source["status"] == "completed_native_alias_pair", "Native source incomplete")
        require(calibration["status"] == "completed_calibrated_same_gpu_paired_forecast_analysis"
                and calibration["calibration"]["all_passed"] is True, "Cached forecast calibration incomplete")
        require(all(calibration["first_query_comparison"][k] for k in
                    ("context_visual_exact", "context_proprio_exact", "goal_features_exact")), "Prior initial equality failed")
        official_path = ROOT / "vendor/jepa-wms/configs/evals/simu_env_planning/mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml"
        official = yaml.safe_load(official_path.read_text())
        objective_cfg = official["planner"]["planning_objective"]
        require(objective_cfg["objective_type"] == "L2" and objective_cfg["alpha"] == .1
                and objective_cfg["sum_all_diffs"] is False, "Native objective config changed")
        require(official["planner"]["horizon"] == 6 and official["planner"]["num_act_stepped"] == 3, "Native horizons changed")
        objective_path = ROOT / protocol["scoring"]["native_source"]
        objectives = module_from_file("dino_cached_native_objectives", objective_path)
        pre_path = ROOT / "vendor/jepa-wms/app/plan_common/datasets/preprocessor.py"
        pre_class = module_from_file("dino_cached_native_preprocessor", pre_path).Preprocessor
        stats = source["strict_load"]["normalization"]
        pre = pre_class(**{key: torch.tensor(stats[key], dtype=torch.float32) for key in
                          ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")}, transform=None)
        records = []
        paths = [native / "summary.json", diagnostic / "summary.json", official_path, objective_path, pre_path]
        for origin in CANDIDATES:
            cache_path = native / origin / "query1_forecast_cache.npz"
            goal_path = native / origin / "goal_encoding.npz"
            prediction_path = diagnostic / f"{origin}_query1.npz"
            trace_path = native / origin / "trace.jsonl"
            paths.extend((cache_path, goal_path, prediction_path, trace_path))
            cache, goal, forecast = (arrays(p, np) for p in (cache_path, goal_path, prediction_path))
            query = source["arms"][origin]["queries"][0]
            require(query["index"] == 1 and query["horizon"] == 6 and query["executed_controls"] == 15,
                    "Query1 definition changed")
            require(query["status"] == "completed", "Query1 incomplete")
            rows = [read_line for line in trace_path.read_text().splitlines()
                    if (read_line := json.loads(line))["planning_call"] == 1]
            require(len(rows) == 15 and [r["policy_step"] for r in rows] == list(range(1, 16))
                    and [r["action_in_chunk"] for r in rows] == list(range(15)), "Actual prefix alignment differs")
            original = torch.from_numpy(cache["mean_original"].copy())
            require(original.shape == (6, 1, 20) and original.dtype == torch.float32, "Mean schema differs")
            original_before = original.clone()
            raw = pre.denormalize_actions(original.reshape(6, 1, 5, 4))
            request = original.reshape(6, 1, 5, 4).clone()
            mask = torch.zeros_like(request, dtype=torch.bool)
            mask[..., :3] = raw[..., :3].abs() > 1
            for axis in range(3):
                for side in (-1, 1):
                    selected = mask[..., axis] & (raw[..., axis].sign() == side)
                    request[..., axis][selected] = source["canonical_boundary_values"][f"axis{axis}_sign{side}"]
            mapped = pre.denormalize_actions(request)
            require(torch.equal(request[~mask], original.reshape_as(request)[~mask]), "Untouched coordinates changed")
            require(torch.equal(mapped[..., :3].clamp(-1, 1), raw[..., :3].clamp(-1, 1))
                    and torch.equal(mapped[..., 3], raw[..., 3]), "Alias physical control semantics differ")
            own = original if origin == CANDIDATES[0] else request.reshape_as(original)
            require(np.array_equal(own.numpy(), cache["mean_request"]), "Own request differs from source")
            require(np.array_equal(original[:3, 0].numpy(), np.asarray(query["normalized_macro_actions"], np.float32)), "Returned macro mean differs")
            executed = raw[:3].reshape(15, 4).numpy()
            require(np.array_equal(executed, np.asarray(query["physical_actions"], np.float32))
                    and np.array_equal(executed, np.asarray([r["action"] for r in rows], np.float32)), "Mean differs from actual controls")
            require(np.array_equal(forecast["actual_control_offsets"], [5, 10, 15])
                    and np.array_equal(cache["actual_control_offsets"], [5, 10, 15]), "Truth offsets changed")
            for key, tail in (("visual", (1, 16, 16, 384)), ("proprio", (256, 20))):
                shape = (1, 1, *tail)
                require(goal[key].shape == cache["context_" + key].shape == shape, "Context/goal schema differs")
                require(goal[key].dtype == cache["context_" + key].dtype == np.float32, "Context/goal dtype differs")
                require(forecast["truth_" + key].shape == (3, 1, *tail), "Actual truth schema differs")
                for representation in REPRESENTATIONS:
                    value = forecast[representation + "_" + key]
                    require(value.shape == (7, 1, *tail) and value.dtype == np.float32
                            and np.isfinite(value).all(), "Forecast schema or values differ")
                    require(np.array_equal(value[:1], cache["context_" + key]), "Forecast context changed")
                require(np.isfinite(goal[key]).all(), "Nonfinite goal")
            successes = [r["policy_step"] for r in rows if r["success"]]
            utility = {"actual_controls": 15, "cumulative_reward_15": math.fsum(float(r["reward"]) for r in rows),
                       "terminal_reward_15": float(rows[-1]["reward"]), "terminal_reach_distance_15": float(rows[-1]["reach_distance"]),
                       "terminal_success_15": bool(rows[-1]["success"]), "ever_success_15": bool(successes),
                       "first_success_within_15": successes[0] if successes else None,
                       "successful_controls_within_15": len(successes)}
            records.append(dict(origin=origin, cache=cache, goal=goal, forecast=forecast, actions=original,
                                actions_before=original_before, utility=utility))
            report["checks"][origin] = {"request_and_execution_checks_passed": True,
                                        "saturated_xyz_coordinates_in_full_mean": int(mask.sum()),
                                        "full_original_mean_shape": list(original.shape),
                                        "actual_truth_control_offsets": [5, 10, 15]}
        for key in FEATURES:
            require(np.array_equal(records[0]["goal"][key], records[1]["goal"][key]), "Cross-candidate goal differs")
            require(np.array_equal(records[0]["cache"]["context_" + key], records[1]["cache"]["context_" + key]), "Cross-candidate context differs")
        report["checks"].update(shared_context_and_goal_exact=True, prior_all_calibration_passed=True,
                                 original_means_exactly_equal=bool(torch.equal(records[0]["actions"], records[1]["actions"])))
        target = TensorDict({key: torch.from_numpy(records[0]["goal"][key].copy()) for key in FEATURES})
        target_before = {key: target[key].clone() for key in FEATURES}
        objective = objectives.ReprTargetDistMPCObjective(official, target_enc=target, alpha=.1, sum_all_diffs=False)
        costs, rankings = {}, {}
        with torch.inference_mode():
            for representation in REPRESENTATIONS:
                cost = []
                for record in records:
                    enc = TensorDict({key: torch.from_numpy(record["forecast"][representation + "_" + key].copy()) for key in FEATURES})
                    value = objective(enc, record["actions"], keepdims=False)
                    report["activity"]["native_objective_calls"] += 1
                    require(value.shape == (1,) and torch.isfinite(value).all(), "Native cost not one finite scalar")
                    cost.append(float(value.item()))
                costs[representation] = cost
                order = sorted(range(2), key=lambda i: (cost[i], i))
                rankings[representation] = {"candidate_indices_best_first": order,
                    "candidate_names_best_first": [CANDIDATES[i] for i in order],
                    "exact_cost_tie": cost[0] == cost[1], "cost_candidate1_minus_candidate0": cost[1] - cost[0],
                    "selected_candidate_index": order[0], "selected_candidate": CANDIDATES[order[0]],
                    "selected_actual_prefix_utility": records[order[0]]["utility"]}
        require(all(torch.equal(r["actions"], r["actions_before"]) for r in records), "Original action tensor mutated")
        require(all(torch.equal(target[k], target_before[k]) for k in FEATURES), "Goal tensor mutated")
        first = rankings["original"]["selected_candidate_index"]
        second = rankings["canonical"]["selected_candidate_index"]
        delta = {}
        for key, value in records[first]["utility"].items():
            other = records[second]["utility"][key]
            delta[key] = None if value is None or other is None else other - value
        report.update(status="completed_cached_two_mean_ranking", sources=[str(p.relative_to(ROOT)) for p in paths],
                      cost_matrix={"rows": list(REPRESENTATIONS), "columns": list(CANDIDATES),
                                   "values": [costs[r] for r in REPRESENTATIONS], "lower_is_better": True,
                                   "native_formula": protocol["scoring"]["formula"], "terminal_control_horizon": 30},
                      rankings=rankings, candidate_actual_utilities={r["origin"]: r["utility"] for r in records},
                      selected_utility_contrast={"ranking_flip": first != second,
                          "direction": "canonical representation selection minus original representation selection",
                          "actual_truth_horizon_controls": 15, "differences": delta,
                          "does_not_estimate": "30-control oracle outcomes, full CEM ranking changes, closed-loop mediation or attack success rate"})
        report["checks"].update(original_actions_unchanged=True, original_goal_unchanged=True,
                                 only_objective_called_no_model_loaded=True)
        report["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__, "device": "cpu",
                             "dtype": "float32", "threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized()}
        require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    except Exception as error:
        failed = True
        report.update(status="invalid_or_missing_source_cache", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        with out.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "cost_matrix", "candidate_actual_utilities", "selected_utility_contrast", "activity", "elapsed_seconds")}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT, help="New output JSON; existing files are never overwritten")
    main(parser.parse_args().out.resolve())
