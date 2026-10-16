#!/usr/bin/env python3
"""One bounded cached-input GRASP batching and native AR-sync cost probe."""
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
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/jepa_grasp_cached_batch_profile_20261016.json"


class ResourceDeferred(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run(config_path, output):
    started = time.monotonic()
    protocol = json.loads(config_path.read_text())
    native = json.loads((ROOT / protocol["native_protocol"]).read_text())
    out = (output or ROOT / protocol["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    (out / "launched_native_protocol.json").write_text(json.dumps(native, indent=2) + "\n")
    planner_path = ROOT / native["planner_source"]
    (out / "launched_planner.py").write_bytes(planner_path.read_bytes())
    activity = {key: 0 for key in protocol["max_activity"]}
    completed = activity.copy()
    report = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path), "purpose": protocol["purpose"],
              "activity": activity, "completed_activity": completed, "stages": [],
              "sync_steps": [], "gpu_samples": [], "checkpoint_files": [],
              "task_utility": None, "clean_gate": None, "attack_success": None,
              "environment_instantiations": 0, "input_action_optimizer_updates": 0,
              "estimate": None, "active": None}
    deadline = started + protocol["resource"]["hard_wall_seconds"]
    phase = None
    torch = None
    exit_code = 0

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        path = out / "summary.json.tmp"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        path.replace(out / "summary.json")

    def time_check():
        if time.monotonic() >= deadline:
            raise ResourceDeferred("Frozen 60-second total budget exhausted")

    def alarm(*_):
        raise ResourceDeferred("Frozen runtime alarm or external termination")

    def enter(key, amount=1):
        time_check()
        require(activity[key] + amount <= protocol["max_activity"][key], "Activity cap: " + key)
        activity[key] += amount

    def finish(key, amount=1):
        completed[key] += amount
        time_check()

    def forbid(*_, **__):
        activity["network_attempts"] += 1
        raise RuntimeError("Network prohibited in cached local probe")

    def memory_gate(admission=False):
        time_check()
        raw = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=uuid,memory.free,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        uuid, free, total = [part.strip() for part in raw.strip().split(",")]
        if report["gpu_samples"]:
            require(uuid == report["gpu_samples"][0]["uuid"], "GPU UUID changed")
        report["gpu_samples"].append({"uuid": uuid, "free_mib": int(free),
                                       "total_mib": int(total), "admission": admission,
                                       "elapsed_seconds": time.monotonic() - started})
        threshold = protocol["resource"]["admission_free_mib" if admission else "boundary_free_mib"]
        if int(free) < threshold:
            raise ResourceDeferred("Shared GPU free memory below fixed boundary")
        persist()

    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    persist()
    try:
        require(protocol["stages"] == [0, 1, 49, 50] and protocol["sync_stage"] == 50
                and protocol["sync_repeats"] == 3 and protocol["threads"] == 4
                and protocol["resource"]["hard_wall_seconds"] == 60, "Frozen probe changed")
        require(native["planner"]["horizon"] == 6 and native["planner"]["opt_steps"] == 500
                and native["planner"]["gd_interval"] == native["planner"]["gd_opt_steps"] == 50
                and native["planner"]["schedule_decay"]
                and native["planner"]["init_goal_weight"] == 5
                and native["planner"]["min_goal_weight"] == 0, "Native schedule changed")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        memory_gate(admission=True)
        import numpy as np
        import torch
        from tensordict import TensorDict
        import pilot_dino_metaworld_clean_gate as helper
        import pilot_jepa_metaworld_reach_wall as wall
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        require(torch.cuda.is_available(), "CUDA unavailable; no fallback")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.30, 0)
        torch.backends.cudnn.benchmark = True
        report["runtime"] = {"python": sys.version, "torch": torch.__version__,
                             "numpy": np.__version__, "threads": torch.get_num_threads(),
                             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                             "device": "cuda:0", "precision": "float32", "memory_fraction": .30}
        cached = ROOT / protocol["cached_run"]
        old = json.loads((cached / "summary.json").read_text())
        require(old["status"] == "resource_deferred", "Prior run is not terminal resource-deferred")
        heartbeat = old["episodes"][0]["queries"][0]["latest_heartbeat"]
        require(heartbeat["completed_grasp_loss_evaluations"] > max(protocol["stages"]),
                "Saved stages lack native consumption evidence")
        directory = cached / protocol["cached_episode"]
        with np.load(directory / "native_virtual_context.npz", allow_pickle=False) as data:
            context = {key: data[key].copy() for key in data.files}
        require(tuple(context["visual_shape"]) == (1, 16, 16, 384)
                and context["s_0_flat"].shape == context["g_flat"].shape == (1, 98304)
                and context["initial_proprio"].shape == (1, 1, 256, 16), "Cached context shape differs")
        saved = {}
        for stage in protocol["stages"]:
            with np.load(directory / f"native_virtual_input_k{stage}.npz", allow_pickle=False) as data:
                saved[stage] = {key: data[key].copy() for key in data.files}
            require(saved[stage]["s_flat"].shape == (1, 5, 98304)
                    and saved[stage]["actions"].shape == (1, 6, 20), "Cached stage shape differs")
        for array in [context[k] for k in ("s_0_flat", "g_flat", "initial_proprio")] + [
                value for inputs in saved.values() for value in inputs.values()]:
            require(array.dtype == np.float32 and np.isfinite(array).all(), "Nonfinite/non-FP32 cached input")
        spec = importlib.util.spec_from_file_location("profile_native_grasp", planner_path)
        planner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(planner_module)
        original_load = torch.load
        allowed = {wall.MODEL.resolve(), helper.DINO_WEIGHTS.resolve()}
        loaded = set()

        def local_load(path, *args, **kwargs):
            resolved = Path(path).resolve()
            require(resolved in allowed and resolved not in loaded, "Unexpected/repeated checkpoint load")
            enter("checkpoint_loads")
            loaded.add(resolved)
            report["checkpoint_files"].append({"path": str(resolved), "bytes": resolved.stat().st_size})
            value = original_load(path, *args, **kwargs)
            finish("checkpoint_loads")
            return value

        def rng_state():
            return (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
                    torch.cuda.get_rng_state(0).clone())

        def rng_equal(before, after):
            return {"python": before[0] == after[0],
                    "numpy": before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
                             and before[1][2:] == after[1][2:],
                    "torch_cpu": torch.equal(before[2], after[2]),
                    "torch_cuda": torch.equal(before[3], after[3])}

        with ExitStack() as guards:
            for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                                (socket, "create_connection"), (torch.hub, "download_url_to_file")):
                guards.enter_context(patch.object(owner, name, side_effect=forbid))
            guards.enter_context(patch.object(torch, "load", side_effect=local_load))
            guards.enter_context(patch.object(helper, "MODEL", wall.MODEL))
            report["status"] = "loading_local_models"
            persist()
            model, _, report["strict_load"] = helper.load_model(wall.cfg_for(helper, out, "cuda"), "cuda")
            require(loaded == allowed and all(not p.requires_grad for p in model.parameters()), "Load/freeze differs")
            report["setup_and_load_seconds"] = time.monotonic() - started
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            root = torch.from_numpy(context["s_0_flat"]).to("cuda")
            goal = torch.from_numpy(context["g_flat"]).to("cuda")
            prop = torch.from_numpy(context["initial_proprio"]).to("cuda")
            shape = tuple(int(x) for x in context["visual_shape"])
            z_init = TensorDict({"visual": root.view(1, 1, *shape), "proprio": prop}, device="cuda")
            planner = planner_module.GRASPPlanner(unroll=model.unroll, action_dim=model.action_dim,
                                                  decode_unroll=None, **native["planner"])

            def pred_pre(*_):
                require(phase in ("native", "batch", "sync"), "Unexpected predictor call")
                enter("predictor_forwards")
                enter("sync_predictor_forwards" if phase == "sync" else "one_step_predictor_forwards")
                enter("sync_semantic_transitions" if phase == "sync" else "one_step_semantic_transitions",
                      6 if phase == "batch" else 1)

            def pred_post(*_):
                finish("predictor_forwards")
                finish("sync_predictor_forwards" if phase == "sync" else "one_step_predictor_forwards")
                finish("sync_semantic_transitions" if phase == "sync" else "one_step_semantic_transitions",
                       6 if phase == "batch" else 1)

            guards.callback(model.model.predictor.register_forward_pre_hook(pred_pre).remove)
            guards.callback(model.model.predictor.register_forward_hook(pred_post).remove)
            for module, key in ((model.model.encoder, "visual_encoder_forwards"),
                                (model.model.proprio_encoder, "proprio_encoder_forwards")):
                guards.callback(module.register_forward_pre_hook(lambda *_, k=key: enter(k)).remove)
            original_unroll = model.unroll

            def unroll(*args, **kwargs):
                enter("unroll_calls")
                value = original_unroll(*args, **kwargs)
                finish("unroll_calls")
                return value

            planner.unroll = unroll
            guards.enter_context(patch.object(model, "encode", side_effect=lambda *_a, **_k: enter("encode_calls")))
            original_backward = torch.Tensor.backward

            def backward(tensor, *args, **kwargs):
                enter("backward_calls")
                value = original_backward(tensor, *args, **kwargs)
                finish("backward_calls")
                return value

            guards.enter_context(patch.object(torch.Tensor, "backward", backward))

            def measured(call):
                time_check()
                before = rng_state()
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                tick = time.monotonic()
                result = call()
                end.record()
                end.synchronize()
                elapsed = time.monotonic() - tick
                unchanged = rng_equal(before, rng_state())
                require(all(unchanged.values()), "Compute changed RNG state")
                time_check()
                return result, {"wall_seconds": elapsed, "cuda_seconds": begin.elapsed_time(end) / 1000,
                                "rng_unchanged": unchanged}

            def compare(left, right):
                left, right = left.detach().cpu(), right.detach().cpu()
                require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), "Nonfinite result")
                diff = right.double() - left.double()
                norm = float(left.double().norm())
                return {"shape": list(left.shape), "bitwise_equal": torch.equal(left, right),
                        "allclose": torch.allclose(left, right, atol=protocol["comparison"]["diagnostic_atol"],
                                                    rtol=protocol["comparison"]["diagnostic_rtol"]),
                        "max_abs_difference": float(diff.abs().max()),
                        "rms_difference": float(diff.square().mean().sqrt()),
                        "relative_l2_difference": float(diff.norm()) / norm if norm else None,
                        "native_l2": norm}

            report["status"] = "profiling_cached_inputs"
            for stage in protocol["stages"]:
                memory_gate()
                report["active"] = {"stage": stage, "phase": "native_then_batch"}
                persist()
                state_input = torch.from_numpy(saved[stage]["s_flat"]).to("cuda")
                action_input = torch.from_numpy(saved[stage]["actions"]).to("cuda")
                weight = 5.0 + (0.0 - 5.0) * ((stage % 50) / 49)
                results, timings = {}, {}
                for mode in ("native", "batch"):
                    phase = mode
                    state = state_input.detach().clone().requires_grad_(True)
                    action = action_input.detach().clone().requires_grad_(True)

                    def compute():
                        if mode == "native":
                            predictions = []
                            original_single = planner._single_step_rollout_flat

                            def record(*args, **kwargs):
                                pred = original_single(*args, **kwargs)
                                predictions.append(pred)
                                return pred

                            with patch.object(planner, "_single_step_rollout_flat", side_effect=record):
                                loss = planner._compute_grasp_loss(state, action, goal, root, 1, shape, z_init,
                                                                   goal_weight=weight)
                            prediction = torch.stack(predictions, dim=1)
                        else:
                            full = torch.cat([root.unsqueeze(1), state, goal.unsqueeze(1)], dim=1)
                            prediction = planner._single_step_rollout_flat(
                                full[:, :-1].detach().reshape(6, -1), action.reshape(6, 1, 20),
                                6, shape, z_init).reshape(1, 6, -1)
                            dyn = [((prediction[:, t] - full[:, t + 1]) ** 2).mean() for t in range(6)]
                            shaping = [((prediction[:, t] - goal) ** 2).mean() for t in range(6)]
                            loss = sum(dyn) + weight * sum(shaping)
                        loss.backward()
                        return {"predictions": prediction.detach(), "loss": loss.detach(),
                                "action_gradient": action.grad.detach(), "state_gradient": state.grad.detach()}

                    result, timings[mode] = measured(compute)
                    results[mode] = {key: value.cpu() for key, value in result.items()}
                    del result, state, action
                equality = {key: compare(results["native"][key], results["batch"][key]) for key in results["native"]}
                require(torch.equal(state_input.cpu(), torch.from_numpy(saved[stage]["s_flat"]))
                        and torch.equal(action_input.cpu(), torch.from_numpy(saved[stage]["actions"])),
                        "Original cached inputs changed")
                row = {"stage": stage, "goal_weight": weight, "timings": timings, "comparisons": equality,
                       "original_inputs_unchanged": True,
                       "native_loss": float(results["native"]["loss"]),
                       "batched_loss": float(results["batch"]["loss"])}
                report["stages"].append(row)
                np.savez(out / f"stage_{stage}_results.npz", **{
                    mode + "_" + key: value.numpy() for mode, values in results.items() for key, value in values.items()})
                persist()
                del results, state_input, action_input

            phase = "sync"
            action_input = torch.from_numpy(saved[50]["actions"]).to("cuda")
            for repeat in range(3):
                memory_gate()
                report["active"] = {"phase": "native_H6_sync", "repeat": repeat}
                persist()
                with patch.object(planner, "gd_opt_steps", 1):
                    updated, timing = measured(lambda: planner._sync_step(root, goal, action_input, 1, shape, z_init))
                require(bool(torch.isfinite(updated).all()), "Nonfinite sync actions")
                report["input_action_optimizer_updates"] += 1
                report["sync_steps"].append({"repeat": repeat, "timing": timing,
                                              "max_action_update": float((updated - action_input).abs().max())})
                np.save(out / f"sync_update_{repeat}.npy", updated.cpu().numpy(), allow_pickle=False)
                require(torch.equal(action_input.cpu(), torch.from_numpy(saved[50]["actions"])),
                        "Native sync changed its original cached action input")
                persist()
                del updated
            require(activity == completed, "Some attempted calls did not complete")
            for key, cap in protocol["max_activity"].items():
                require(activity[key] == cap, "Unexpected final activity count: " + key)
            batch_times = [row["timings"]["batch"]["wall_seconds"] for row in report["stages"]]
            native_times = [row["timings"]["native"]["wall_seconds"] for row in report["stages"]]
            sync_times = [row["timing"]["wall_seconds"] for row in report["sync_steps"]]
            batch_mean, sync_mean = statistics.mean(batch_times), statistics.mean(sync_times)
            planning = 3 * (37 / 6) * (500 * batch_mean + 500 * sync_mean)
            old_first_loss = old["episodes"][0]["queries"][0].get("latest_heartbeat", {}).get("elapsed_seconds")
            report["estimate"] = {
                "kind": "small_sample_shared_GPU_cost_estimate_not_task_execution",
                "batch_loss_mean_seconds": batch_mean, "native_loss_mean_seconds": statistics.mean(native_times),
                "native_H6_sync_update_mean_seconds": sync_mean,
                "independent_loss_speedup_ratio": statistics.mean(native_times) / batch_mean,
                "three_root_planning_seconds": planning,
                "three_root_planning_seconds_min_max_components": [
                    3 * (37 / 6) * 500 * (min(batch_times) + min(sync_times)),
                    3 * (37 / 6) * 500 * (max(batch_times) + max(sync_times))],
                "probe_observed_setup_and_load_seconds": report["setup_and_load_seconds"],
                "planning_plus_probe_setup_seconds": planning + report["setup_and_load_seconds"],
                "prior_run_total_seconds": old.get("elapsed_seconds"),
                "prior_latest_loss_heartbeat_elapsed_field": old_first_loss,
                "assumptions": ["H5/H2 query cost scales linearly from H6; not measured here",
                                "Four cached loss stages and three fresh-Adam sync updates represent later optimizer work; unproved",
                                "No warmup discarded; first-call overhead and current shared-GPU contention are included",
                                "Loss timing includes forward/backward but excludes main optimizer update and noise",
                                "Sync timing includes fresh Adam initialization/update; later 49 updates reuse optimizer state",
                                "Physical setup, encoding, final rollout and task/report overhead are additional unless explicitly reported"],
                "does_not_authorize_new_task_or_budget": True}
            report["all_bitwise_equal"] = all(item["bitwise_equal"] for row in report["stages"] for item in row["comparisons"].values())
            report["all_fixed_allclose_checks"] = all(item["allclose"] for row in report["stages"] for item in row["comparisons"].values())
            report["status"] = "completed_cached_batch_and_native_sync_profile"
    except ResourceDeferred as exc:
        report.update(status="resource_deferred", reason=str(exc))
    except Exception as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        report.update(status="resource_deferred" if oom else "failed", reason=str(exc),
                      error_type=type(exc).__name__, traceback=traceback.format_exc())
        exit_code = 0 if oom else 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if torch is not None and torch.cuda.is_initialized():
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["active"] = None
        report["budget_overrun_seconds"] = max(0.0, time.monotonic() - deadline)
        persist()
    print(json.dumps({"status": report["status"], "summary": str(out / "summary.json"),
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.config.resolve(), args.output))
