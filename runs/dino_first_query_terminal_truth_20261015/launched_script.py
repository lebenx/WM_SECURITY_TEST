#!/usr/bin/env python3
"""Encode five cached observations at most; no predictor, planner or environment."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dino_first_query_terminal_truth_20261015.json"
NAMES = ("clean_identity_copy", "canonical_request")
KEYS = ("visual", "proprio")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class CalibrationFailure(RuntimeError):
    pass


class ResourceDeferred(RuntimeError):
    pass


def main(config):
    protocol = read(config)
    out, final = ROOT / protocol["out_dir"], ROOT / protocol["report"]
    require(not out.exists() and not final.exists(), "Refusing existing output directory/report")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(config.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    counts = {"native_encodes": 0, "calibration_encodes": 0, "terminal_truth_encodes": 0,
              "predictor_calls": 0, "unroll_calls": 0, "planner_calls": 0,
              "environment_steps": 0, "api_calls": 0, "training_updates": 0, "native_objective_calls": 0}
    report = {"status": "checking_qualified_replay", "protocol": str(config.relative_to(ROOT)),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "activity": counts,
              "checkpoint_loads": {}, "calibration": {"all_passed": None, "checks": {}},
              "terminal_forecast_errors": None, "true_future_objective": None,
              "physical_utilities": None, "limits": protocol["metrics"]["limits"]}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - start
        write(out / "summary.json", report)
    def expired(*_):
        raise ResourceDeferred("Frozen180-second wall limit reached")
    def forbidden_network(*_args, **_kwargs):
        counts["api_calls"] += 1
        raise RuntimeError("Network prohibited")
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    persist()
    try:
        import numpy as np
        from analyze_dino_native_alias_forecast_gpu import load_npz, numerical_comparison
        require(protocol["selection"]["candidate_order"] == list(NAMES) and protocol["selection"]["query"] == 1,
                "Frozen candidate selection changed")
        native = ROOT / protocol["native_source_run"]
        diagnostic = ROOT / protocol["forecast_source_run"]
        replay_dir = ROOT / protocol["replay_source_run"]
        replay = read(replay_dir / "summary.json")
        old_diag = read(diagnostic / "summary.json")
        require(replay["status"] == protocol["replay_interface"]["completed_status"], "Replay not completed")
        require(all(replay["reconstruction"][key] is True for key in protocol["replay_interface"]["reconstruction_checks"]),
                "Replay reconstruction not qualified")
        require(old_diag["status"] == "completed_calibrated_same_gpu_paired_forecast_analysis"
                and old_diag["calibration"]["all_passed"] is True, "Prior same-GPU diagnostic incomplete")
        old_protocol = read(ROOT / "configs/dino_native_alias_forecast_gpu_20261015.json")
        for key in ("atol", "rtol", "max_absolute_error", "rms_absolute_error"):
            require(protocol["calibration"][key] == old_protocol["calibration"][key], "Calibration tolerance changed")
        rank = read(ROOT / "reports/dino_first_query_two_mean_ranking_20261015.json")
        require(rank["status"] == "completed_cached_two_mean_ranking", "Prior bank score missing")
        setup = load_npz(native / "setup/setup.npz", np)
        bank = load_npz(replay_dir / "physical_actions.npz", np)
        records = []
        require(len(replay["candidates"]) == 2, "Unexpected replay candidate count")
        for i, name in enumerate(NAMES):
            item = replay["candidates"][i]
            require(item["name"] == name and item["status"] == "completed30" and item["controls"] == 30
                    and item["historical_prefix_exact_steps"] == 15 and item["last_prefix_mismatches"] == [],
                    "Candidate replay incomplete/unqualified")
            cache = load_npz(native / name / "query1_forecast_cache.npz", np)
            forecast = load_npz(diagnostic / f"{name}_query1.npz", np)
            goal = load_npz(native / name / "goal_encoding.npz", np)
            require(np.array_equal(bank["mean_original"][i], cache["mean_original"]), "Replayed original mean changed")
            endpoints = {}
            for offset in (15, 30):
                expected = replay_dir / name / f"endpoint{offset}.npz"
                require((ROOT / item[f"endpoint{offset}"]).resolve() == expected.resolve(), "Unexpected endpoint path")
                endpoint = load_npz(expected, np)
                require(int(endpoint["control_offset"]) == offset, "Endpoint offset differs")
                require(endpoint["pixels"].shape == (1, 3, 224, 224) and endpoint["pixels"].dtype == np.uint8
                        and endpoint["proprio"].shape == (1, 4) and endpoint["proprio"].dtype == np.float32,
                        "Endpoint input schema differs")
                require(np.array_equal(endpoint["target"], setup["target"]), "Original physical target changed")
                endpoints[offset] = endpoint
            require(np.array_equal(endpoints[15]["pixels"], cache["actual_pixels"][2])
                    and np.array_equal(endpoints[15]["proprio"], cache["actual_proprio"][2]), "15-control input differs")
            require(np.array_equal(forecast["actual_control_offsets"], [5, 10, 15]), "Prior truth offsets differ")
            for key in KEYS:
                for rep in ("original", "canonical"):
                    require(forecast[rep + "_" + key].shape[0] == 7
                            and np.array_equal(forecast[rep + "_" + key][:1], cache["context_" + key]), "Forecast context/horizon differs")
            records.append(dict(name=name, cache=cache, goal=goal, forecast=forecast,
                                endpoints=endpoints, utility=item["utility30"]))
        for key in KEYS:
            require(np.array_equal(records[0]["goal"][key], records[1]["goal"][key])
                    and np.array_equal(records[0]["cache"]["context_" + key], records[1]["cache"]["context_" + key]),
                    "Shared goal/context changed")
        report["dependency_checks"] = {"completed_replay": True, "exact_source_reconstruction": True,
            "both_historical_prefixes_exact": True, "mean_goal_context_unchanged": True, "actual15_inputs_exact": True}
        report["physical_utilities"] = {r["name"]: r["utility"] for r in records}
        report["sources"] = {"native": str(native.relative_to(ROOT)), "forecasts": str(diagnostic.relative_to(ROOT)),
                             "real_replay": str(replay_dir.relative_to(ROOT)), "ranking": "reports/dino_first_query_two_mean_ranking_20261015.json"}
        from pilot_dino_native_alias_pair import gpu_info
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["memory.free"] < protocol["runtime"]["minimum_free_mib"]:
            raise ResourceDeferred("Actual GPU0 free below3072MiB; no model loaded")
        import torch
        from tensordict import TensorDict
        torch.set_num_threads(4)
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        total = torch.cuda.get_device_properties(0).total_memory
        cap = int(protocol["runtime"]["allocator_cap_gib"] * 1024**3)
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, MODEL, DINO_WEIGHTS
        original_load = torch.load
        def once(path, *args, **kwargs):
            key = str(Path(path).resolve())
            require(key in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and key not in report["checkpoint_loads"], "Unexpected/repeated checkpoint load")
            report["checkpoint_loads"][key] = 1
            return original_load(path, *args, **kwargs)
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=forbidden_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=forbidden_network))
            guards.enter_context(patch.object(torch, "load", side_effect=once))
            report["status"] = "loading_existing_gpu_encoder"; persist()
            model, _, report["strict_load"] = load_model(cfg_for(out, "cuda"), "cuda")
            require(len(report["checkpoint_loads"]) == 2, "Expected two local checkpoint loads")
            def forbidden_predictor(*_):
                counts["predictor_calls"] += 1
                raise RuntimeError("Predictor forward prohibited")
            def forbidden_unroll(*_args, **_kwargs):
                counts["unroll_calls"] += 1
                raise RuntimeError("Unroll prohibited")
            hook = model.model.predictor.register_forward_pre_hook(forbidden_predictor)
            guards.callback(hook.remove)
            guards.enter_context(patch.object(model, "unroll", side_effect=forbidden_unroll))
            def encode(label, pixels, proprio, phase):
                require(counts["native_encodes"] < 5, "Encode budget exceeded")
                counts["native_encodes"] += 1; counts[phase] += 1
                result = model.encode({"visual": torch.from_numpy(pixels[None].copy()),
                                       "proprio": torch.from_numpy(proprio[None].copy())}, act=label != "goal_calibration")
                arrays = {key: result[key].cpu().numpy().copy() for key in KEYS}
                require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values()), "Invalid encoded feature")
                np.savez_compressed(out / f"{label}.npz", **arrays)
                return arrays
            def calibrate(label, pixels, proprio, reference):
                actual = encode(label, pixels, proprio, "calibration_encodes")
                checks = {key: numerical_comparison(actual[key], reference[key], protocol["calibration"], np) for key in KEYS}
                report["calibration"]["checks"][label] = checks; persist()
                if not all(c["passed"] for c in checks.values()):
                    raise CalibrationFailure("Frozen encoder calibration failed: " + label)
            with torch.inference_mode():
                report["status"] = "calibrating_goal_and_both15_endpoints"; persist()
                calibrate("goal_calibration", setup["goal_pixels"], setup["goal_proprio"], records[0]["goal"])
                for r in records:
                    e = r["endpoints"][15]
                    calibrate(r["name"] + "_control15", e["pixels"], e["proprio"],
                              {key: r["forecast"]["truth_" + key][2:3] for key in KEYS})
                require(counts["calibration_encodes"] == 3 and counts["terminal_truth_encodes"] == 0, "Calibration order violated")
                report["calibration"]["all_passed"] = True
                report["status"] = "encoding_two30_control_true_endpoints"; persist()
                for r in records:
                    e = r["endpoints"][30]
                    r["truth"] = encode(r["name"] + "_control30", e["pixels"], e["proprio"], "terminal_truth_encodes")
                    persist()
            torch.cuda.synchronize()
        objective_path = ROOT / "vendor/jepa-wms/evals/simu_env_planning/planning/planning/objectives.py"
        spec = importlib.util.spec_from_file_location("dino_terminal_native_objectives", objective_path)
        objectives = importlib.util.module_from_spec(spec); spec.loader.exec_module(objectives)
        target = TensorDict({key: torch.from_numpy(records[0]["goal"][key].copy()) for key in KEYS})
        objective = objectives.ReprTargetDistMPCObjective(None, target_enc=target, alpha=.1, sum_all_diffs=False)
        errors, costs = {}, []
        with torch.inference_mode():
            for r in records:
                errors[r["name"]] = {}
                for rep in ("original", "canonical"):
                    error = {}
                    for key in KEYS:
                        delta = r["forecast"][rep + "_" + key][6:7].astype(np.float64) - r["truth"][key].astype(np.float64)
                        error[key + "_mse"] = float(np.mean(delta * delta))
                    error["weighted_mse"] = error["visual_mse"] + .1 * error["proprio_mse"]
                    errors[r["name"]][rep] = error
                enc = TensorDict({key: torch.from_numpy(r["truth"][key].copy()) for key in KEYS})
                cost = objective(enc, torch.from_numpy(r["cache"]["mean_original"].copy()))
                counts["native_objective_calls"] += 1
                require(cost.shape == (1,) and torch.isfinite(cost).all(), "Invalid terminal native objective")
                costs.append(float(cost.item()))
        order = sorted(range(2), key=lambda i: (costs[i], i))
        selected, other = order[0], order[1]
        report["terminal_forecast_errors"] = errors
        report["true_future_objective"] = {"candidate_order": list(NAMES), "costs": costs,
            "rank_best_first": [NAMES[i] for i in order], "exact_tie": costs[0] == costs[1],
            "selected_candidate": NAMES[selected], "control_horizon": 30,
            "formula": "Native CPU FP32 terminal visual MSE +0.1*proprio MSE versus unchanged goal",
            "selected_actual_utility": records[selected]["utility"],
            "selects_lower_actual_cumulative_reward": records[selected]["utility"]["cumulative_native_reward"] < records[other]["utility"]["cumulative_native_reward"],
            "selects_lower_actual_terminal_reward": records[selected]["utility"]["terminal_native_reward"] < records[other]["utility"]["terminal_native_reward"],
            "selected_minus_other_actual_cumulative_reward": records[selected]["utility"]["cumulative_native_reward"] - records[other]["utility"]["cumulative_native_reward"],
            "selected_minus_other_actual_terminal_reward": records[selected]["utility"]["terminal_native_reward"] - records[other]["utility"]["terminal_native_reward"]}
        report["cached_forecast_rankings"] = rank["rankings"]
        report["cached_forecast_cost_matrix"] = rank["cost_matrix"]
        report["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__, "device": "cuda:0",
            "feature_dtype": "float32", "objective_device": "cpu", "threads": torch.get_num_threads(),
            "allocator_cap_bytes": cap, "allocator_excludes_context": True,
            "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": True}
        require(counts["native_encodes"] == 5 and counts["terminal_truth_encodes"] == 2
                and counts["predictor_calls"] == counts["unroll_calls"] == 0, "Call budget differs")
        report["status"] = "completed_calibrated_two_mean_terminal_truth"
    except CalibrationFailure as error:
        report.update(status="invalid_numerical_calibration", error=str(error))
        report["calibration"]["all_passed"] = False
    except ResourceDeferred as error:
        report.update(status="resource_deferred", error=str(error))
    except BaseException as error:
        oom = type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()
        report.update(status="resource_deferred" if oom else "invalid_dependency_or_runtime", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
        require(not final.exists(), "Refusing result report overwrite")
        write(final, report)
    print(json.dumps({key: report[key] for key in ("status", "activity", "terminal_forecast_errors", "true_future_objective", "elapsed_seconds")}))
    return 0 if report["status"] == "completed_calibrated_two_mean_terminal_truth" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    raise SystemExit(main(parser.parse_args().config.resolve()))
