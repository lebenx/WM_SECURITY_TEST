#!/usr/bin/env python3
"""Bounded official-GRASP/released-JEPA reach-wall Clean research composite."""
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
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/jepa_grasp_native_clean_20261016.json"


class ResourceDeferred(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap():
    target = ROOT / ".deps/dino_mujoco330"
    if os.environ.get("JEPA_GRASP_BOOTSTRAPPED") != "1":
        os.environ.update(JEPA_GRASP_BOOTSTRAPPED="1", CUDA_VISIBLE_DEVICES="0",
                          MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LIBGL_ALWAYS_SOFTWARE="1",
                          PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        os.environ["PYTHONPATH"] = str(target) + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Physical CUDA mapping changed")
    sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
    sys.path.insert(0, str(target))


def gpu_sample():
    output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid,memory.free,memory.total",
                                      "--format=csv,noheader,nounits"], text=True, timeout=10)
    uuid, free, total = [part.strip() for part in output.strip().split(",")]
    return {"uuid": uuid, "free_mib": int(free), "total_mib": int(total)}


def run(config_path, output):
    protocol = json.loads(config_path.read_text())
    out = (output or ROOT / protocol["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    counts = {key: 0 for key in protocol["max_activity"]}
    completed = {key: 0 for key in counts}
    report = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": protocol["purpose"], "protocol": str(config_path), "source_commit": protocol["source_commit"],
              "activity": counts, "completed_activity": completed, "episodes": [], "gate": None,
              "checkpoint_files": [], "gpu_samples": [], "active": None,
              "unmeasured_roots": protocol["roots"].copy(), "model_api_calls": 0,
              "training_updates": 0, "first_query_profile": None}
    active_episode, env = None, None
    deadline = started + protocol["resource"]["hard_wall_seconds"]
    query_deadline = None

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        temp = out / "summary.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(out / "summary.json")

    def time_check(*_):
        now = time.monotonic()
        if now >= deadline:
            raise ResourceDeferred("Total 1200-second wall budget exhausted")
        if query_deadline is not None and now >= query_deadline:
            raise ResourceDeferred("First query exceeded its frozen 180-second cap")

    def alarm(*_):
        time_check()
        raise ResourceDeferred("Frozen wall-clock alarm fired")

    def arm_timer():
        stop_at = min(deadline, query_deadline) if query_deadline is not None else deadline
        signal.setitimer(signal.ITIMER_REAL, max(0.001, stop_at - time.monotonic()))

    def enter(key):
        time_check()
        counts[key] += 1
        require(counts[key] <= protocol["max_activity"][key], "Activity cap exceeded: " + key)

    def finish(key):
        completed[key] += 1
        time_check()

    def forbid(*_, **__):
        counts["network_attempts"] += 1
        raise RuntimeError("Network access forbidden")

    def memory_gate(admission=False):
        time_check()
        state = gpu_sample()
        state.update(elapsed_seconds=time.monotonic() - started, admission=admission)
        if report["gpu_samples"]:
            require(state["uuid"] == report["gpu_samples"][0]["uuid"], "GPU UUID changed")
        report["gpu_samples"].append(state)
        persist()
        threshold = protocol["resource"]["admission_free_mib" if admission else "query_boundary_free_mib"]
        if state["free_mib"] < threshold:
            raise ResourceDeferred("Insufficient free VRAM at fixed query boundary")

    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(ResourceDeferred("External termination")))
    arm_timer()
    persist()
    exit_code = 0
    try:
        require(protocol["roots"] == [1, 2, 3] and protocol["threads"] == 4
                and protocol["planner"]["opt_steps"] == 500 and protocol["planner"]["horizon"] == 6
                and protocol["planner"]["num_act_stepped"] == 3, "Frozen configuration changed")
        memory_gate(admission=True)
        import numpy as np
        import torch
        import mujoco
        from tensordict import TensorDict
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
        import evals.simu_env_planning.envs.init as env_init
        import pilot_dino_metaworld_clean_gate as helper
        import pilot_jepa_metaworld_reach_wall as wall
        require(Path(mujoco.__file__).resolve().is_relative_to(wall.MUJOCO_TARGET)
                and mujoco.__version__ == mujoco.mj_versionString() == "3.3.0", "MuJoCo runtime differs")
        require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Renderer differs")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        require(torch.cuda.is_available(), "CUDA unavailable; no backend fallback")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(protocol["resource"]["cuda_memory_fraction"], 0)
        torch.backends.cudnn.benchmark = True
        report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                              "numpy": np.__version__, "mujoco": mujoco.__version__, "mujoco_path": mujoco.__file__,
                              "device": "cuda:0", "threads": 4, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "allocator_fraction": protocol["resource"]["cuda_memory_fraction"]}
        cpu = json.loads((ROOT / protocol["cpu_summary"]).read_text())
        cpu_audit = json.loads((ROOT / protocol["cpu_analysis"]).read_text())
        require([e["seed"] for e in cpu["episodes"]] == [1, 2, 3]
                and cpu_audit["status"] == "completed_independent_reach_wall_cpu_audit", "CPU baseline panel differs")
        require(cpu_audit["final_success_counts"]["wall_aware_servo"] == 1
                and cpu_audit["final_success_counts"]["noop_repeat0"] == 0, "Frozen baseline success counts differ")
        report["baseline_sources"] = {"cpu_summary": protocol["cpu_summary"], "cpu_analysis": protocol["cpu_analysis"],
                                       "final_success_counts": cpu_audit["final_success_counts"]}
        planner_module = load_module("frozen_jepa_grasp_planner", ROOT / protocol["planner_source"])
        (out / "launched_planner.py").write_bytes((ROOT / protocol["planner_source"]).read_bytes())
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

        with ExitStack() as guards:
            for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                                (socket, "create_connection"), (torch.hub, "download_url_to_file")):
                guards.enter_context(patch.object(owner, name, side_effect=forbid))
            guards.enter_context(patch.object(torch, "load", side_effect=local_load))
            guards.enter_context(patch.object(helper, "MODEL", wall.MODEL))
            cfg = wall.cfg_for(helper, out, "cuda")
            report["status"] = "loading_local_models"
            persist()
            model, pre, report["strict_load"] = helper.load_model(cfg, "cuda")
            require(loaded == allowed and all(not p.requires_grad for p in model.parameters()), "Model load/freeze differs")
            for module, key in ((model.model.encoder, "visual_encoder_forwards"),
                                (model.model.proprio_encoder, "proprio_encoder_forwards"),
                                (model.model.predictor, "predictor_forwards")):
                guards.callback(module.register_forward_pre_hook(lambda *_, k=key: enter(k)).remove)
                guards.callback(module.register_forward_hook(lambda *_, k=key: finish(k)).remove)
            original_encode, original_unroll = model.encode, model.unroll

            def encoded(*args, **kwargs):
                enter("encode_calls")
                value = original_encode(*args, **kwargs)
                finish("encode_calls")
                return value

            def unrolled(*args, **kwargs):
                enter("unroll_calls")
                value = original_unroll(*args, **kwargs)
                finish("unroll_calls")
                return value

            guards.enter_context(patch.object(model, "encode", side_effect=encoded))
            guards.enter_context(patch.object(model, "unroll", side_effect=unrolled))
            planner = planner_module.GRASPPlanner(unroll=model.unroll, action_dim=model.action_dim,
                                                   decode_unroll=None, **protocol["planner"])
            require(planner.device == torch.device("cuda") and model.action_dim == 20, "Planner device/action contract differs")
            original_grasp_loss = planner._compute_grasp_loss
            loss_entries = 0

            def recorded_grasp_loss(s_flat, actions, g_flat, s_0_flat, batch, vis_shape, z_init, **kwargs):
                nonlocal loss_entries
                k = loss_entries
                loss_entries += 1
                first_query = active_episode["seed"] == 1 and query["index"] == 1
                if first_query and k in (0, 1, 49, 50):
                    np.savez_compressed(directory / f"native_virtual_input_k{k}.npz",
                                        s_flat=s_flat.detach().cpu().numpy(), actions=actions.detach().cpu().numpy())
                    if k == 0:
                        np.savez_compressed(directory / "native_virtual_context.npz",
                                            s_0_flat=s_0_flat.detach().cpu().numpy(),
                                            g_flat=g_flat.detach().cpu().numpy(),
                                            initial_proprio=z_init["proprio"].detach().cpu().numpy(),
                                            visual_shape=np.asarray(vis_shape))
                    query.setdefault("native_virtual_input_snapshots", []).append(k)
                value = original_grasp_loss(s_flat, actions, g_flat, s_0_flat, batch, vis_shape, z_init, **kwargs)
                if loss_entries % 50 == 0:
                    heartbeat = {"seed": active_episode["seed"], "query": query["index"],
                                 "completed_grasp_loss_evaluations": loss_entries,
                                 "query_elapsed_seconds": time.monotonic() - tick,
                                 "predictor_forwards": counts["predictor_forwards"]}
                    query["latest_heartbeat"] = heartbeat
                    persist()
                    print(json.dumps({"heartbeat": heartbeat}), flush=True)
                return value

            guards.enter_context(patch.object(planner, "_compute_grasp_loss", side_effect=recorded_grasp_loss))
            original_make_env = env_init.make_env

            def feature_stats(z):
                result = {}
                for key in ("visual", "proprio"):
                    value = z[key].detach()
                    require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "Nonfinite or non-FP32 features")
                    result[key] = {"shape": list(value.shape), "max_abs": float(value.abs().max()),
                                   "rms": float(value.double().square().mean().sqrt())}
                return result

            for seed, reference in zip(protocol["roots"], cpu["episodes"]):
                memory_gate()
                directory = out / f"seed{seed}_clean"
                active_episode = {"seed": seed, "status": "preparing", "queries": [], "policy_controls_executed": 0,
                                  "matched_cpu_outcomes": {b["condition"]: b["outcome"] for b in reference["branches"]}}
                report["episodes"].append(active_episode)
                report["active"] = {"seed": seed, "phase": "setup"}
                report["status"] = "running"
                report["unmeasured_roots"].remove(seed)
                persist()
                created = []
                setup_phase = True

                def tracked_make_env(*args, **kwargs):
                    instance = original_make_env(*args, **kwargs)
                    role = "main" if not created else "expert"
                    created.append(instance)
                    original_step = instance.step

                    def step(action, *args, **kwargs):
                        kind = "expert_steps" if role == "expert" else ("warmup_steps" if setup_phase else "policy_steps")
                        enter("environment_steps")
                        enter(kind)
                        result = original_step(action, *args, **kwargs)
                        completed["environment_steps"] += 1
                        completed[kind] += 1
                        time_check()
                        return result
                    instance.step = step
                    return instance

                try:
                    with patch.object(env_init, "make_env", side_effect=tracked_make_env):
                        env, obs, goal, setup = helper.setup_episode(cfg, seed, directory)
                    setup_phase = False
                    require(len(created) == 2, "Unexpected setup environment count")
                    checks = {b["condition"]: wall.setup_checks(setup, b["setup"]) for b in reference["branches"]}
                    active_episode["setup"] = setup
                    active_episode["setup_checks"] = checks
                    require(all(all(v.values()) for v in checks.values()), "Initial/goal/expert setup differs from cached roots")
                    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
                    goal_enc = model.encode(goal.unsqueeze(0), act=False).detach()
                    active_episode["goal_features"] = feature_stats(goal_enc)
                    objective = ReprTargetDistMPCObjective(cfg, target_enc=goal_enc, **cfg.planner.planning_objective)
                    planner.set_objective(objective)
                    rows = []
                    with (directory / "trace.jsonl").open("x") as trace:
                        while env.steps_left() > 0:
                            memory_gate()
                            index = len(active_episode["queries"]) + 1
                            steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                            horizon = min(6, steps_left)
                            query = {"index": index, "status": "planning", "elapsed_before": env.elapsed_steps(),
                                     "planning_steps_left": steps_left, "horizon": horizon, "executed_control_count": 0,
                                     "activity_before": counts.copy()}
                            active_episode["queries"].append(query)
                            report["active"] = {"seed": seed, "phase": "planning", "query": index}
                            persist()
                            rng_before = {"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state(0)}
                            tick = time.monotonic()
                            loss_entries = 0
                            if report["first_query_profile"] is None:
                                query_deadline = min(deadline, tick + protocol["resource"]["first_complete_query_cap_seconds"])
                                arm_timer()
                            encoded_obs = model.encode(obs.unsqueeze(0), act=True)
                            query["observation_features"] = feature_stats(encoded_obs)
                            enter("planner_calls")
                            with torch.enable_grad():
                                result = planner.plan(encoded_obs, steps_left=steps_left)
                            finish("planner_calls")
                            torch.cuda.synchronize()
                            elapsed_query = time.monotonic() - tick
                            time_check()
                            query_deadline = None
                            arm_timer()
                            normalized = result.actions.detach().cpu()
                            loss = result.losses.detach().cpu()
                            require(normalized.shape == (min(3, horizon), 20) and normalized.dtype == torch.float32
                                    and bool(torch.isfinite(normalized).all()) and loss.shape == (500, 1)
                                    and bool(torch.isfinite(loss).all()), "Invalid actions/losses")
                            forecast = result.predicted_best_encs_over_iterations[-1]
                            query["final_forecast_features"] = feature_stats(forecast)
                            actions = pre.denormalize_actions(normalized.reshape(-1, 4))
                            require(actions.dtype == torch.float32 and bool(torch.isfinite(actions).all()), "Invalid physical controls")
                            query.update(status="executing", elapsed_seconds=elapsed_query,
                                         normalized_macro_actions=normalized.tolist(), physical_actions=actions.tolist(),
                                         losses=loss.tolist(), planned_control_count=len(actions), activity_after=counts.copy(),
                                         cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved())
                            query["activity_delta"] = {k: counts[k] - query["activity_before"][k] for k in counts}
                            require(query["activity_delta"]["predictor_forwards"] == 1001 * horizon
                                    and query["activity_delta"]["unroll_calls"] == 500 * horizon + 501,
                                    "Native GRASP query call budget differs")
                            torch.save({"normalized_macro_actions": normalized, "physical_actions": actions, "losses": loss,
                                        "forecast": {k: forecast[k].detach().cpu() for k in ("visual", "proprio")},
                                        "rng_before": rng_before, "rng_after": {"cpu": torch.random.get_rng_state(),
                                                                                "cuda": torch.cuda.get_rng_state(0)}},
                                       directory / f"query{index}.pt")
                            helper.write(directory / "queries.json", active_episode["queries"])
                            persist()
                            report["active"]["phase"] = "executing"
                            for local_index, action in enumerate(actions):
                                visual, info, row = helper.trace_step(env, action, index, local_index)
                                row["executor_xyz_clip_float32"] = action[:3].clamp(-1, 1).tolist()
                                row["executor_gripper_requested"] = float(action[3])
                                rows.append(row)
                                trace.write(json.dumps(row, allow_nan=False) + "\n"); trace.flush()
                                query["executed_control_count"] += 1
                                active_episode["policy_controls_executed"] = len(rows)
                                obs = TensorDict({"visual": visual, "proprio": info["proprio"]})
                                if row["done"]:
                                    break
                            query["status"] = "completed"
                            helper.write(directory / "queries.json", active_episode["queries"])
                            if report["first_query_profile"] is None:
                                remaining_estimate = elapsed_query * (37 / 6 - 1)
                                remaining_budget = deadline - time.monotonic()
                                report["first_query_profile"] = {"seed": seed, "query": index, "seconds": elapsed_query,
                                    "executed_policy_controls": len(rows), "estimated_root_planning_seconds": elapsed_query * 37 / 6,
                                    "estimated_remaining_root_planning_seconds": remaining_estimate,
                                    "remaining_total_wall_seconds": remaining_budget,
                                    "rule": "Fixed source-derived horizon sum scaling; estimate, not a measured full-root runtime"}
                                persist()
                                if remaining_estimate > remaining_budget:
                                    raise ResourceDeferred("First-query estimate cannot finish this root within remaining wall budget")
                            persist()
                            print(json.dumps({"seed": seed, "query": index, "policy_controls": len(rows),
                                              "reach_distance": rows[-1]["reach_distance"], "success": rows[-1]["success"],
                                              "query_seconds": elapsed_query}), flush=True)
                            if rows[-1]["done"]:
                                break
                    require([q["elapsed_before"] for q in active_episode["queries"]] == protocol["physical_protocol"]["query_elapsed_steps"]
                            and [q["horizon"] for q in active_episode["queries"]] == protocol["physical_protocol"]["query_horizons"]
                            and [q["executed_control_count"] for q in active_episode["queries"]] == protocol["physical_protocol"]["executed_control_counts"],
                            "Physical execution schedule differs")
                    active_episode["outcome"] = helper.outcome(rows, directory)
                    active_episode["status"] = "completed"
                    report["active"] = None
                    persist()
                    failed = sum(e["status"] == "completed" and not e["outcome"]["final_success"] for e in report["episodes"])
                    if failed >= 2:
                        report["status"] = "completed_futility_stop"
                        report["gate"] = {"passed": False, "reason": "Two completed roots failed; at least2/3 gate is mathematically unreachable",
                                          "completed_roots": len(report["episodes"]), "completed_failures": failed,
                                          "required_final_successes": 2,
                                          "unmeasured_roots": report["unmeasured_roots"].copy()}
                        break
                finally:
                    for created_env in created:
                        try:
                            created_env.close()
                        except Exception:
                            pass
                    env = None
                    persist()
            if report["gate"] is None:
                require(len(report["episodes"]) == 3 and all(e["status"] == "completed" for e in report["episodes"]), "Incomplete panel")
                successes = sum(bool(e["outcome"]["final_success"]) for e in report["episodes"])
                require(successes >= 2, "Failed fixed panel was not caught by futility rule")
                report["status"] = "completed_clean_qualification"
                report["gate"] = {"passed": True, "final_successes": successes, "denominator": 3,
                                  "wall_aware_final_successes": 1, "native_minus_wall_aware_successes_descriptive": successes - 1,
                                  "scope": protocol["gate"]["scope"]}
            require(counts == completed, "Incomplete calls in a completed result")
    except ResourceDeferred as exc:
        report.update(status="resource_deferred", reason=str(exc), gate=None)
        if active_episode is not None and active_episode["status"] != "completed":
            active_episode["status"] = "partial_resource_deferred"
    except BaseException as exc:
        report.update(status="error", exception_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), gate=None)
        if active_episode is not None and active_episode["status"] != "completed":
            active_episode["status"] = "technical_error"
        exit_code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["active_at_stop"] = report["active"]
        report["active"] = None
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        persist()
    print(json.dumps({"status": report["status"], "gate": report["gate"], "activity": counts,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bootstrap()
    raise SystemExit(run(args.config.resolve(), args.output))
