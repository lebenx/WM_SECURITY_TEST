#!/usr/bin/env python3
"""Separate same-GPU cached-mean diagnostic after a failed CPU calibration."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
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
CONFIG = ROOT / "configs/dino_native_alias_forecast_gpu_20261015.json"
ORIGINS = ("clean_identity_copy", "canonical_request")
FEATURES = ("visual", "proprio")


class CalibrationFailure(RuntimeError):
    pass


class ResourceDeferred(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_npz(path, np):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def numerical_comparison(actual, reference, limits, np):
    require(actual.shape == reference.shape, "Calibration shape differs")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "Nonfinite calibration tensor")
    def metrics(a, b):
        delta = a.astype(np.float64) - b.astype(np.float64)
        maximum = float(np.abs(delta).max())
        rms = float(np.sqrt(np.mean(delta * delta)))
        close = bool(np.allclose(a, b, atol=limits["atol"], rtol=limits["rtol"]))
        return {"max_absolute_error": maximum, "rms_absolute_error": rms,
                "reference_rms": float(np.sqrt(np.mean(b.astype(np.float64) ** 2))),
                "allclose": close,
                "passed": close and maximum <= limits["max_absolute_error"] and rms <= limits["rms_absolute_error"]}
    result = metrics(actual, reference)
    result["per_timepoint"] = [metrics(actual[i], reference[i]) for i in range(len(actual))]
    result["passed"] = result["passed"] and all(x["passed"] for x in result["per_timepoint"])
    return result


def canonical_request(original, pre, boundaries, torch):
    request = original.reshape(*original.shape[:2], 5, 4).clone()
    raw = pre.denormalize_actions(request)
    require(torch.isfinite(raw).all(), "Nonfinite raw mean")
    mask = torch.zeros_like(request, dtype=torch.bool)
    mask[..., :3] = raw[..., :3].abs() > 1
    for axis in range(3):
        for side in (-1, 1):
            selected = mask[..., axis] & (raw[..., axis].sign() == side)
            request[..., axis][selected] = boundaries[axis, side]
    mapped = pre.denormalize_actions(request)
    require(torch.equal(request[~mask], original.reshape_as(request)[~mask]), "Untouched normalized coordinate changed")
    require(torch.equal(mapped[..., :3].clamp(-1, 1), raw[..., :3].clamp(-1, 1)), "Effective XYZ differs")
    require(torch.equal(mapped[..., 3], raw[..., 3]), "Raw gripper differs")
    return request.reshape_as(original).clone()


def read_sources(source, protocol, np):
    previous = json.loads((source / "summary.json").read_text())
    require(previous["status"] == "completed_native_alias_pair", "Native pair incomplete")
    require(previous["decision"]["consider_separate_recovery_pilot"] is True, "Frozen source trigger not met")
    require(previous["activity"]["unroll_calls"] == 420 and previous["activity"]["policy_controls"] == 198, "Native pair budget differs")
    selection = protocol["selection"]
    require(selection["trajectory_origins"] == list(ORIGINS) and selection["queries_per_origin"] == list(range(1, 8)), "Frozen query panel changed")
    records, goals = [], {}
    for origin in ORIGINS:
        arm = previous["arms"][origin]
        require(arm["status"] == "completed" and len(arm["queries"]) == 7, "Incomplete arm")
        goals[origin] = load_npz(source / origin / "goal_encoding.npz", np)
        traces = [json.loads(line) for line in (source / origin / "trace.jsonl").read_text().splitlines()]
        require(len(traces) == 99, "Unexpected policy trace length")
        for qi, query in enumerate(arm["queries"], 1):
            horizon, offsets = selection["horizons"][qi - 1], selection["actual_offsets"][qi - 1]
            require(query["index"] == qi and query["status"] == "completed" and query["horizon"] == horizon, "Query status/horizon differs")
            require(query["actual_forecast_aligned_control_offsets"] == offsets, "Actual offset schema changed")
            cached = load_npz(source / origin / f"query{qi}_forecast_cache.npz", np)
            count = len(offsets)
            shapes = {"context_visual": (1, 1, 1, 16, 16, 384), "context_proprio": (1, 1, 256, 20),
                      "mean_original": (horizon, 1, 20), "mean_request": (horizon, 1, 20),
                      "mean_forecast_visual": (horizon + 1, 1, 1, 16, 16, 384),
                      "mean_forecast_proprio": (horizon + 1, 1, 256, 20),
                      "actual_pixels": (count, 1, 3, 224, 224), "actual_proprio": (count, 1, 4),
                      "actual_control_offsets": (count,)}
            for key, shape in shapes.items():
                dtype = np.uint8 if key == "actual_pixels" else np.int64 if key == "actual_control_offsets" else np.float32
                require(cached[key].shape == shape and cached[key].dtype == dtype and np.isfinite(cached[key]).all(), "Cache schema differs: " + key)
            require(np.array_equal(cached["actual_control_offsets"], offsets), "Cached offsets differ")
            own = "original" if origin == ORIGINS[0] else "canonical"
            rows = [r for r in traces if r["planning_call"] == qi]
            expected = 15 if qi < 7 else 9
            require(len(rows) == query["executed_controls"] == expected, "Executed prefix length differs")
            require([r["action_in_chunk"] for r in rows] == list(range(expected)), "Trace action order differs")
            require(np.array_equal(np.asarray(query["normalized_macro_actions"], np.float32), cached["mean_original"][:3, 0]), "Returned mean differs from query")
            for key in FEATURES:
                require(np.array_equal(cached["mean_forecast_" + key][0:1], cached["context_" + key]), "Forecast prefix differs from context")
            records.append({"origin": origin, "query": query, "qi": qi, "H": horizon, "offsets": offsets,
                            "cache": cached, "own": own, "rows": rows})
    for key, shape in (("visual", (1, 1, 1, 16, 16, 384)), ("proprio", (1, 1, 256, 20))):
        require(all(goals[o][key].shape == shape and goals[o][key].dtype == np.float32 and np.isfinite(goals[o][key]).all() for o in ORIGINS), "Goal schema differs")
        require(np.array_equal(goals[ORIGINS[0]][key], goals[ORIGINS[1]][key]), "Task goal encodings differ")
    first = {"context_" + k + "_exact": bool(np.array_equal(records[0]["cache"]["context_" + k], records[7]["cache"]["context_" + k])) for k in FEATURES}
    require(all(first.values()), "First-query contexts differ across origins")
    first.update(goal_features_exact=True,
                 full_original_mean_exact=bool(np.array_equal(records[0]["cache"]["mean_original"], records[7]["cache"]["mean_original"])),
                 executed_prefix_exact=bool(np.array_equal(np.asarray([r["action"] for r in records[0]["rows"]], np.float32), np.asarray([r["action"] for r in records[7]["rows"]], np.float32))),
                 interpretation="Shared first context does not pair the two different selected action sequences or their true futures. Representation effects are evaluated within each query.")
    return previous, records, first


def main(config_path, override_out):
    protocol = json.loads(config_path.read_text())
    out = (override_out or ROOT / protocol["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    source = ROOT / protocol["source_run"]
    started = time.monotonic()
    counts = {"unroll_calls": 0, "own_condition_unrolls": 0, "counterfactual_unrolls": 0,
              "predictor_forwards": 0, "native_encodes": 0, "encoder_calibration_checks": 0,
              "environment_steps": 0, "planner_calls": 0, "network_attempts": 0, "training_updates": 0}
    report = {"status": "checking_cached_sources", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path), "source_run": str(source), "known_outcome_scope": protocol["known_outcome_scope"],
              "disclosure": protocol["disclosure"], "prior_cpu_attempt": protocol["prior_cpu_attempt"],
              "projection_limits": protocol["metrics"]["projection_limits"],
              "activity": counts, "checkpoint_loads": {}, "calibration": {"limits": protocol["calibration"], "all_passed": None},
              "queries": [], "paired_summary": None, "interpretation": protocol["metrics"]["causal_limit"]}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        write(out / "summary.json", report)
    def expired(*_):
        raise TimeoutError("Frozen300-second same-GPU diagnostic wall limit reached")
    def forbidden_network(*_args, **_kwargs):
        counts["network_attempts"] += 1
        raise RuntimeError("No network allowed")
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    persist()
    try:
        from pilot_dino_native_alias_pair import gpu_info
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["memory.free"] < protocol["runtime"]["minimum_start_free_mib"]:
            raise ResourceDeferred("Startup actual free VRAM below5632MiB; no model loaded")
        import numpy as np
        import torch
        from tensordict import TensorDict
        torch.set_num_threads(protocol["runtime"]["threads"])
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        total = torch.cuda.get_device_properties(0).total_memory
        cap = int(protocol["runtime"]["allocator_cap_gib"] * 1024**3)
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        previous, records, first = read_sources(source, protocol, np)
        report.update(source_decision=previous["decision"], first_query_comparison=first,
                      runtime={"interpreter": sys.executable, "torch": torch.__version__, "torch_path": torch.__file__,
                               "device": "cuda:0", "dtype": "float32", "threads": torch.get_num_threads(),
                               "allocator_cap_bytes": cap, "allocator_fraction": cap / total, "allocator_excludes_context": True,
                               "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": True})
        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, MODEL, DINO_WEIGHTS
        from pilot_dino_native_alias_pair import canonical_boundaries
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
            report["status"] = "loading_same_gpu_model"; persist()
            model, pre, report["strict_load"] = load_model(cfg_for(out, "cuda"), "cuda")
            require(len(report["checkpoint_loads"]) == 2, "Expected two existing checkpoint files")
            def count_predictor(*_):
                require(counts["predictor_forwards"] < 148, "Predictor budget exceeded")
                counts["predictor_forwards"] += 1
            hook = model.model.predictor.register_forward_pre_hook(count_predictor)
            guards.callback(hook.remove)
            boundaries = canonical_boundaries(pre, torch)
            require({f"axis{a}_sign{s}": float(v) for (a, s), v in boundaries.items()} == previous["canonical_boundary_values"], "Canonical boundary values changed")
            by_id = {(r["origin"], r["qi"]): r for r in records}
            for r in records:
                c = r["cache"]
                original = torch.from_numpy(c["mean_original"].copy())
                r["requests"] = {"original": original, "canonical": canonical_request(original, pre, boundaries, torch)}
                require(np.array_equal(r["requests"][r["own"]].numpy(), c["mean_request"]), "Saved own-condition request differs")
                raw = pre.denormalize_actions(original[:3, 0].reshape(-1, 4)).numpy()
                require(np.array_equal(raw, np.asarray(r["query"]["physical_actions"], np.float32)), "Native denormalization differs from planned controls")
                require(np.array_equal(raw[:len(r["rows"])], np.asarray([row["action"] for row in r["rows"]], np.float32)), "Native mean does not match executed controls")
                r["z"] = TensorDict({k: torch.from_numpy(c["context_" + k].copy()).to("cuda:0") for k in FEATURES})
                r["result"] = {"origin": r["origin"], "query": r["qi"], "horizon": r["H"], "offsets": r["offsets"],
                               "own_representation": r["own"], "status": "pending_calibration", "own_forecast_calibration": None,
                               "endpoint_encoder_calibration": None, "paired_errors": None}
                report["queries"].append(r["result"])
            def forecast(r, representation, phase):
                require(counts["unroll_calls"] < 28, "Forecast budget exceeded")
                actions = r["requests"][representation].to("cuda:0")
                before = actions.clone()
                counts["unroll_calls"] += 1; counts[phase] += 1
                prediction = model.unroll(r["z"], act_suffix=actions)
                require(torch.equal(actions, before), "Forecast request mutated")
                arrays = {}
                for k in FEATURES:
                    value = prediction[k].cpu().numpy().copy()
                    require(value.shape == r["cache"]["mean_forecast_" + k].shape and np.isfinite(value).all(), "Forecast schema/value differs")
                    require(np.array_equal(value[0:1], r["cache"]["context_" + k]), "Forecast initial prefix changed")
                    require(np.array_equal(r["z"][k].cpu().numpy(), r["cache"]["context_" + k]), "Context mutated")
                    arrays[k] = value
                return arrays
            limits = protocol["calibration"]
            with torch.inference_mode():
                # No counterfactual forecast occurs until every factual/encoder gate passes.
                report["status"] = "calibrating_all_own_conditions"; persist()
                for r in records:
                    own = forecast(r, r["own"], "own_condition_unrolls")
                    r["predictions"] = {r["own"]: own}
                    r["file"] = out / f"{r['origin']}_query{r['qi']}.npz"
                    np.savez_compressed(r["file"], **{r["own"] + "_" + k: v for k, v in own.items()})
                    checks = {k: numerical_comparison(own[k], r["cache"]["mean_forecast_" + k], limits, np) for k in FEATURES}
                    r["result"]["own_forecast_calibration"] = checks
                    persist()
                    if not all(c["passed"] for c in checks.values()):
                        raise CalibrationFailure(f"Own-condition forecast mismatch at {r['origin']} query{r['qi']}")
                    truth = {k: [] for k in FEATURES}
                    for i, offset in enumerate(r["offsets"]):
                        require(counts["native_encodes"] < 38, "Encoder budget exceeded")
                        counts["native_encodes"] += 1
                        encoded = model.encode({"visual": torch.from_numpy(r["cache"]["actual_pixels"][i:i+1].copy()),
                                                "proprio": torch.from_numpy(r["cache"]["actual_proprio"][i:i+1].copy())}, act=True)
                        for k in FEATURES:
                            require(encoded[k].shape == r["z"][k].shape and torch.isfinite(encoded[k]).all(), "Encoded truth schema/value differs")
                            truth[k].append(encoded[k][:, 0].cpu().numpy().copy())
                        # Keep the actual encoded arrays even if the next gate
                        # stops this attempt; no extra encode is needed to audit.
                        np.savez_compressed(r["file"], **{r["own"] + "_" + k: v for k, v in own.items()},
                                            **{"truth_" + k: np.stack(v) for k, v in truth.items()},
                                            actual_control_offsets=np.asarray(r["offsets"][:i + 1], np.int64))
                        if offset == 15 and r["qi"] < 7:
                            next_cache = by_id[r["origin"], r["qi"] + 1]["cache"]
                            checks = {k: numerical_comparison(encoded[k].cpu().numpy(), next_cache["context_" + k], limits, np) for k in FEATURES}
                            counts["encoder_calibration_checks"] += 1
                            r["result"]["endpoint_encoder_calibration"] = checks
                            persist()
                            if not all(c["passed"] for c in checks.values()):
                                raise CalibrationFailure(f"Endpoint encoder mismatch at {r['origin']} query{r['qi']}")
                    r["truth"] = {k: np.stack(v) for k, v in truth.items()}
                    r["result"]["status"] = "own_condition_calibrated"
                    np.savez_compressed(r["file"], **{r["own"] + "_" + k: v for k, v in own.items()},
                                        **{"truth_" + k: v for k, v in r["truth"].items()})
                    persist()
                    print(json.dumps({"phase": "calibration", "origin": r["origin"], "query": r["qi"], "passed": True}), flush=True)
                require(counts["own_condition_unrolls"] == 14 and counts["native_encodes"] == 38 and counts["encoder_calibration_checks"] == 12, "Calibration panel incomplete")
                report["calibration"]["all_passed"] = True
                report["status"] = "computing_same_gpu_counterfactuals"; persist()
                table = []
                for r in records:
                    other = "canonical" if r["own"] == "original" else "original"
                    r["predictions"][other] = forecast(r, other, "counterfactual_unrolls")
                    errors = {}
                    indices = np.asarray(r["offsets"], np.int64) // 5
                    for representation, values in r["predictions"].items():
                        errors[representation] = {}
                        for k in FEATURES:
                            delta = values[k][indices].astype(np.float64) - r["truth"][k].astype(np.float64)
                            errors[representation][k] = (delta * delta).mean(axis=tuple(range(1, delta.ndim)))
                        errors[representation]["weighted"] = errors[representation]["visual"] + .1 * errors[representation]["proprio"]
                    rows = []
                    for j, offset in enumerate(r["offsets"]):
                        actual = r["rows"][offset - 1]
                        row = {"origin": r["origin"], "query": r["qi"], "offset": offset,
                               "environment_step": actual["environment_step"], "actual_reward": actual["reward"],
                               "actual_reach_distance": actual["reach_distance"], "actual_success": actual["success"]}
                        for k in (*FEATURES, "weighted"):
                            row["original_" + k + "_mse"] = float(errors["original"][k][j])
                            row["canonical_" + k + "_mse"] = float(errors["canonical"][k][j])
                            row["canonical_minus_original_" + k + "_mse"] = float(errors["canonical"][k][j] - errors["original"][k][j])
                        rows.append(row); table.append(row)
                    r["result"].update(status="completed", paired_errors=rows,
                                         mean_paired_weighted_difference=float(np.mean(errors["canonical"]["weighted"] - errors["original"]["weighted"])))
                    np.savez_compressed(r["file"], **{representation + "_" + k: v for representation, values in r["predictions"].items() for k, v in values.items()},
                                        **{"truth_" + k: v for k, v in r["truth"].items()}, actual_control_offsets=np.asarray(r["offsets"], np.int64))
                    persist()
                    print(json.dumps({"phase": "paired", "origin": r["origin"], "query": r["qi"], "mean_paired_weighted_difference": r["result"]["mean_paired_weighted_difference"]}), flush=True)
            require(counts["unroll_calls"] == 28 and counts["predictor_forwards"] == 148 and counts["native_encodes"] == 38, "Final model-call budget differs")
            torch.cuda.synchronize()
            with (out / "paired_errors.csv").open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
            report["paired_summary"] = {
                origin: {"queries": 7, "endpoints": 19,
                         "mean_of_within_query_weighted_differences": float(np.mean([r["result"]["mean_paired_weighted_difference"] for r in records if r["origin"] == origin])),
                         "mean_of_within_endpoint_weighted_differences": float(np.mean([r["canonical_minus_original_weighted_mse"] for r in table if r["origin"] == origin])),
                         "scope": "Descriptive paired representation differences along this origin's selected actions; correlated observations, no between-trajectory causal subtraction."}
                for origin in ORIGINS}
            report["status"] = "completed_calibrated_same_gpu_paired_forecast_analysis"
    except CalibrationFailure as error:
        report.update(status="invalid_numerical_calibration", error=str(error))
        report["calibration"]["all_passed"] = False
    except (TimeoutError, ResourceDeferred) as error:
        report.update(status="resource_deferred", error=str(error))
    except BaseException as error:
        oom = type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()
        report.update(status="resource_deferred" if oom else "failed_closed_cached_forecast_analysis", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if not oom:
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if "torch" in locals():
            report["cuda_initialized"] = torch.cuda.is_initialized()
            if report["cuda_initialized"]:
                report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({"status": report["status"], "activity": counts, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    main(args.config.resolve(), args.out)
