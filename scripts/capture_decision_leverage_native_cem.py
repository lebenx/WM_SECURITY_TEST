#!/usr/bin/env python3
"""One fresh native CEM query, observed without changing its computation.

This captures optimization internals, not a candidate-utility experiment.
No alias transformation, extra model evaluation, training, or task rollout.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
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
DEFAULT = ROOT / "configs/decision_leverage_native_capture.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def bootstrap():
    marker = "DECISION_LEVERAGE_NATIVE_CAPTURE_BOOTSTRAPPED"
    if os.environ.get(marker) != "1":
        os.environ.update({marker: "1", "CUDA_VISIBLE_DEVICES": "0",
                           "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                           "LIBGL_ALWAYS_SOFTWARE": "1",
                           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["PYTHONPATH"] = str(ROOT / ".deps/dino_mujoco330") + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Device mapping changed")
    require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Renderer changed")
    sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
    sys.path.insert(0, str(ROOT / ".deps/dino_mujoco330"))


def gpu_info():
    fields = ["memory.total", "memory.free", "memory.used", "utilization.gpu"]
    r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(fields),
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    return dict(zip(fields, [int(v.strip()) for v in r.stdout.strip().split(",")]))


def process_gpu_mib():
    r = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    for line in r.stdout.splitlines():
        pid, memory = [v.strip() for v in line.split(",")]
        if int(pid) == os.getpid():
            return int(memory)
    return 0


def verify(out):
    import numpy as np
    s = json.loads((out / "summary.json").read_text())
    require(s["status"] == "completed_native_query_capture", "Capture did not complete")
    for name, h in s["artifact_sha256"].items():
        require(digest(out / name) == h, "Artifact changed: " + name)
    require(s["activity"]["native_act_calls"] == 1 and s["activity"]["cost_calls"] == 15, "Wrong query/cost count")
    with np.load(out / "final_native_output.npz", allow_pickle=False) as end:
        require(np.array_equal(end["returned_prefix"], end["full_mean"][:3]), "Returned prefix differs")
        require(np.array_equal(end["full_mean"], end["last_elite_mean_gpu"]), "Mean differs from captured GPU reduction")
        with np.load(out / "iteration_14.npz", allow_pickle=False) as last:
            require(np.array_equal(last["actions"], end["final_population"]), "Final population differs")
            require(np.array_equal(last["costs"], end["final_costs"]), "Final costs differ")
            require(np.array_equal(last["actions"][:, last["elite_indices"]], end["elite_actions"]), "Elite action extraction differs")
    previous_mean = None
    for index in range(15):
        with np.load(out / f"iteration_{index:02}.npz", allow_pickle=False) as a:
            require(a["actions"].shape == (6, 300, 20) and a["costs"].shape == (300,), "Wrong captured shapes")
            require(np.isfinite(a["actions"]).all() and np.isfinite(a["costs"]).all(), "Nonfinite captured values")
            require(len(set(a["elite_indices"].tolist())) == 10, "Elite membership is not a ten-element subset")
            selected = a["costs"][a["elite_indices"]]
            require(np.all(selected[:-1] <= selected[1:]), "Elite order differs from ascending cost")
            require(float(selected.max()) <= float(np.partition(a["costs"], 9)[9]), "An elite is above the kth cost")
            if previous_mean is not None:
                require(np.array_equal(a["actions"][:, 0], previous_mean), "Next population omitted the native previous mean")
            else:
                require(np.count_nonzero(a["actions"][:, 0]) == 0, "First included mean is not zero")
            previous_mean = a["elite_mean_gpu"].copy()
    require(all(s["checks"].values()), "A live observation invariant failed")
    return {"status": "verified_cached_capture", "iterations": 15, "native_act_calls": 1,
            "cached_actions": 4500, "physical_candidate_replays": 0,
            "new_model_calls": 0, "new_environment_controls": 0}


def run(config_path):
    protocol = json.loads(config_path.read_text())
    require(protocol["status"] == "frozen_before_single_query_calibration", "Protocol not frozen")
    require(protocol["physical_seed"] == 3000 and protocol["native_local_seed"] == 1, "Seed contract changed")
    out = ROOT / protocol["out_dir"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    env = None
    torch = None
    summary = {"status": "resource_precheck", "pid": os.getpid(),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": digest(config_path), "runner_sha256": digest(__file__),
               "physical_seed": 3000, "native_local_seed": 1,
               "scope": "One native CEM query; no candidate utility, independent replication, E1 completion, attack or defense claim.",
               "activity": {"native_act_calls": 0, "cost_calls": 0, "native_mean_forecasts_retained": 0,
                            "expert_setup_controls": 0, "root_warmup_controls": 0,
                            "policy_execution_controls": 0, "extra_model_evaluations": 0,
                            "network_attempts": 0, "api_calls": 0, "training_updates": 0},
               "checkpoint_loads": {}, "iterations": [], "checks": {}, "artifact_sha256": {},
               "sampled_process_gpu_peak_mib": 0}

    def persist():
        summary["elapsed_seconds"] = time.monotonic() - start
        write(out / "summary.json", summary)

    def expired(*_):
        raise TimeoutError("Frozen 180-second total wall limit reached; no retry")

    def no_network(*_, **__):
        summary["activity"]["network_attempts"] += 1
        raise RuntimeError("Network forbidden for local native capture")

    def check_memory():
        value = process_gpu_mib()
        summary["sampled_process_gpu_peak_mib"] = max(summary["sampled_process_gpu_peak_mib"], value)
        require(value <= protocol["runtime"]["process_vram_budget_mib"], "Observed process VRAM exceeds 8GiB; stop")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    persist()
    try:
        summary["gpu_before_import"] = gpu_info()
        require(summary["gpu_before_import"]["memory.free"] >= 8192, "Startup free VRAM below 8GiB; stop before model load")
        for name, h in protocol["source_sha256"].items():
            require(digest(ROOT / name) == h, "Frozen source changed: " + name)
        for item in protocol["checkpoints"]:
            path = Path(item["path"])
            require(digest(path) == item["sha256"], "Checkpoint digest differs: " + str(path))
        import numpy as np
        import torch as torch_module
        torch = torch_module
        import mujoco
        from omegaconf import OmegaConf
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, setup_episode, physical, MODEL, DINO_WEIGHTS
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        require(mujoco.__version__ == "3.3.0" and str(ROOT / ".deps/dino_mujoco330") in mujoco.__file__, "Pinned MuJoCo overlay missing")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        total = torch.cuda.get_device_properties(0).total_memory
        # Keep one GiB of the process budget outside the allocator for CUDA context.
        cap = 7 * 1024**3
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        summary["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__,
                              "mujoco": mujoco.__version__, "mujoco_file": mujoco.__file__,
                              "allocator_cap_bytes": cap, "process_budget_bytes": 8 * 1024**3,
                              "process_peak_scope": "Sampled after model/setup and each iteration; allocator cap is enforced separately.",
                              "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": True,
                              "threads": 4, "microbatching": False}
        cfg = cfg_for(out, "cuda")
        require(cfg.local_seed == 1, "Native local seed differs")
        # Unused upstream logging paths retain unresolved configuration variables.
        write(out / "effective_native_config.json", OmegaConf.to_container(cfg, resolve=False))
        original_load = torch.load

        def checked_load(path, *args, **kwargs):
            key = str(Path(path).resolve())
            require(key in {str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())}, "Unexpected checkpoint load")
            require(key not in summary["checkpoint_loads"], "Repeated checkpoint load")
            summary["checkpoint_loads"][key] = 1
            return original_load(path, *args, **kwargs)

        with ExitStack() as guards, torch.no_grad():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            guards.enter_context(patch.object(torch, "load", side_effect=checked_load))
            summary["status"] = "loading_local_model"; persist()
            model, pre, summary["strict_model_load"] = load_model(cfg, "cuda")
            require(not model.training and all(not p.requires_grad for p in model.parameters()), "Model not frozen/eval")
            check_memory()
            summary["status"] = "fresh_seed3000_setup"; persist()
            env, initial, goal, summary["setup"] = setup_episode(cfg, 3000, out / "setup")
            summary["activity"].update(expert_setup_controls=100, root_warmup_controls=1)
            physical_before = physical(env)
            raw = env.proprio_env.unwrapped
            require(raw.reward_function_version == "v2", "Unexpected task reward implementation")
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            integration = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, integration, state_spec)
            np.savez(out / "root_integration.npz", integration=integration, site_pos=raw.model.site_pos.copy(),
                     **{"physical_" + k: v for k, v in physical_before.items()})
            agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
            agent.set_goal(goal.clone())
            planner = agent.planner
            require((planner.num_samples, planner.iterations, planner.num_elites, planner.horizon, planner.num_act_stepped) == (300, 15, 10, 6, 3), "Native CEM dimensions changed")
            require(planner.momentum_mean == planner.momentum_std == 0 and planner.max_norms is None and not planner.distribute_planner, "Native CEM update changed")
            require(planner.objective is agent.objective and agent.objective.alpha == .1 and not agent.objective.sum_all_diffs, "Native objective changed")
            goal_ref = {k: v.detach().cpu().clone() for k, v in agent.goal_state_enc.items()}
            np.savez(out / "goal_encoding.npz", **{k: v.numpy() for k, v in goal_ref.items()})
            global_rng_initial = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone())
            local_initial = agent.local_gpu_generator.get_state().clone()
            local_cpu_initial = agent.local_generator.get_state().clone()
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            native_cost = planner.cost_function
            unroll_original = planner.unroll
            objective_original = planner.objective
            context_ref = None
            last = {}

            def rng_snapshot():
                return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                        torch.cuda.get_rng_state(0).clone(), agent.local_gpu_generator.get_state().clone(),
                        agent.local_generator.get_state().clone())

            def rng_equal(a, b):
                return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
                        and a[1][2:] == b[1][2:] and all(torch.equal(x, y) for x, y in zip(a[2:], b[2:])))

            def observe_cost(actions, z_init):
                nonlocal context_ref
                index = summary["activity"]["cost_calls"]
                require(index < 15 and actions.shape == (6, 300, 20) and actions.dtype == torch.float32, "Unexpected native cost call")
                rng_before = rng_snapshot()
                action_copy = actions.detach().cpu().clone()
                z_copy = {k: v.detach().cpu().clone() for k, v in z_init.items()}
                if context_ref is None:
                    context_ref = z_copy
                    np.savez(out / "context_encoding.npz", **{k: v.numpy() for k, v in z_copy.items()})
                require(all(torch.equal(z_copy[k], context_ref[k]) for k in context_ref), "Context changed across CEM iterations")
                require(all(torch.equal(v.detach().cpu(), goal_ref[k]) for k, v in agent.goal_state_enc.items()), "Goal changed")
                if index:
                    require(torch.equal(action_copy[:, 0], last["mean_cpu"]), "Native next population differs from recorded update")
                else:
                    require(torch.count_nonzero(action_copy[:, 0]) == 0, "Initial native mean sample differs")
                require(rng_equal(rng_before, rng_snapshot()), "Pre-call observer consumed RNG")
                costs = native_cost(actions, z_init)
                require(costs.shape == (300,) and costs.dtype == torch.float32 and bool(torch.isfinite(costs).all()), "Native cost output invalid")
                require(torch.equal(actions.detach().cpu(), action_copy), "Native cost mutated caller actions")
                require(all(torch.equal(v.detach().cpu(), z_copy[k]) for k, v in z_init.items()), "Native cost mutated context")
                require(all(torch.equal(v.detach().cpu(), goal_ref[k]) for k, v in agent.goal_state_enc.items()), "Native cost mutated goal")
                require(rng_equal(rng_before, rng_snapshot()), "Cost/observer consumed an undeclared RNG stream")
                costs_copy = costs.detach().cpu().clone()
                elite_indices = torch.topk(-costs, 10, dim=0).indices
                elite_actions = actions[:, elite_indices]
                elite_mean = elite_actions.mean(dim=1)
                mean_cpu = elite_mean.detach().cpu().clone()
                data = {"actions": action_copy.numpy(), "costs": costs_copy.numpy(),
                        "elite_indices": elite_indices.cpu().numpy(), "elite_mean_gpu": mean_cpu.numpy(),
                        "local_cuda_rng_at_cost": rng_before[4].numpy()}
                np.savez(out / f"iteration_{index:02}.npz", **data)
                last.update(actions=action_copy, costs=costs_copy, indices=elite_indices.cpu(),
                            elite_actions=elite_actions.detach().cpu().clone(), mean_cpu=mean_cpu)
                summary["activity"]["cost_calls"] += 1
                summary["iterations"].append({"index": index, "cost_min": float(costs_copy.min()),
                                              "elite_indices": elite_indices.cpu().tolist(),
                                              "capture_file": f"iteration_{index:02}.npz"})
                check_memory(); persist()
                require(torch.equal(costs.detach().cpu(), costs_copy), "Observer modified output costs")
                require(rng_equal(rng_before, rng_snapshot()), "Post-call observer consumed RNG")
                return costs  # Exactly the original tensor object, no replacement or adaptation.

            planner.cost_function = observe_cost
            require(planner.unroll is unroll_original and planner.objective is objective_original, "Observer replaced model/scorer")
            check_memory()
            summary["gpu_before_query"] = gpu_info()
            require(summary["gpu_before_query"]["memory.free"] >= 8192, "Free VRAM fell below 8GiB before native query")
            summary["status"] = "single_native_query_running"; persist()
            summary["activity"]["native_act_calls"] = 1
            query_start = time.monotonic()
            steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
            returned = agent.act(initial.clone(), steps_left=steps_left)
            torch.cuda.synchronize()
            summary["query_seconds"] = time.monotonic() - query_start
            full_mean = planner._prev_mean.detach().cpu().clone()
            require(summary["activity"]["cost_calls"] == 15, "Incomplete native iteration count")
            require(torch.equal(full_mean, last["mean_cpu"]) and torch.equal(returned, full_mean[:3]), "Native returned mean/prefix mismatch")
            require(len(agent._predicted_best_encs_over_iterations) == 15, "Native mean prediction count differs")
            summary["activity"]["native_mean_forecasts_retained"] = 15
            mean_forecasts = {}
            for i, prediction in enumerate(agent._predicted_best_encs_over_iterations):
                for key, value in prediction.items():
                    require(bool(torch.isfinite(value).all()), "Nonfinite native mean forecast")
                    mean_forecasts[f"iteration{i:02}_{key}"] = value.detach().cpu().numpy()
            np.savez(out / "native_mean_forecasts.npz", **mean_forecasts)
            np.savez(out / "final_native_output.npz", full_mean=full_mean.numpy(), returned_prefix=returned.numpy(),
                     final_population=last["actions"].numpy(), final_costs=last["costs"].numpy(),
                     elite_indices=last["indices"].numpy(), elite_actions=last["elite_actions"].numpy(),
                     last_elite_mean_gpu=last["mean_cpu"].numpy(),
                     raw_requested_full_mean=pre.denormalize_actions(full_mean.reshape(-1, 4)).numpy(),
                     raw_requested_final_population=pre.denormalize_actions(last["actions"].permute(1, 0, 2).reshape(300, 30, 4)).numpy(),
                     local_cuda_generator_initial=local_initial.numpy(),
                     local_cuda_generator_final=agent.local_gpu_generator.get_state().numpy(),
                     local_cpu_generator_initial=local_cpu_initial.numpy(),
                     local_cpu_generator_final=agent.local_generator.get_state().numpy(),
                     global_cpu_rng_initial=global_rng_initial[0].numpy(), global_cpu_rng_final=torch.get_rng_state().numpy(),
                     global_cuda_rng_initial=global_rng_initial[1].numpy(), global_cuda_rng_final=torch.cuda.get_rng_state(0).numpy())
            physical_after = physical(env)
            integration_after = np.empty_like(integration)
            mujoco.mj_getState(raw.model, raw.data, integration_after, state_spec)
            require(np.array_equal(integration, integration_after), "Physical integration state changed during query")
            require(all(np.array_equal(v, physical_after[k]) for k, v in physical_before.items()), "Physical root changed during query")
            require(random.getstate() == python_rng and np.array_equal(np.random.get_state()[1], numpy_rng[1]) and np.random.get_state()[2:] == numpy_rng[2:], "Host global RNG changed during native query")
            require(torch.equal(global_rng_initial[0], torch.get_rng_state()) and torch.equal(global_rng_initial[1], torch.cuda.get_rng_state(0)), "Global torch RNG changed during query")
            require(torch.equal(local_cpu_initial, agent.local_generator.get_state()), "Unused native CPU RNG changed")
            require(all(torch.equal(v.detach().cpu(), goal_ref[k]) for k, v in agent.goal_state_enc.items()), "Final goal differs")
            require(planner.unroll is unroll_original and planner.objective is objective_original, "Native model/scorer binding changed")
            check_memory()
            summary["checks"] = {"same_original_cost_tensor_returned": True, "observer_preserves_all_rng_streams": True,
                                  "actions_context_goal_unchanged_during_cost_calls": True,
                                  "all_14_next_population_means_exact": True,
                                  "last_elite_mean_matches_native_full_mean_exact": True,
                                  "returned_three_macro_prefix_exact": True,
                                  "root_state_and_integration_unchanged": True,
                                  "native_goal_objective_and_unroll_preserved": True,
                                  "one_query_no_extra_model_evaluation": True}
            summary["model_call_accounting"] = {"observed_cost_calls": 15, "retained_native_mean_unroll_outputs": 15,
                                                "total_unroll_calls_from_frozen_source_and_observed_structure": 30,
                                                "goal_and_context_encode_calls_from_frozen_source": 2,
                                                "not_independent_model_forward_hook_counts": True}
            summary["status"] = "completed_native_query_capture"
    except BaseException as error:
        summary.update(status="failed_preserved_no_retry", exception_type=type(error).__name__,
                       exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            env.close()
        if torch is not None and torch.cuda.is_initialized():
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "summary.json":
                summary["artifact_sha256"][str(path.relative_to(out))] = digest(path)
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["utility_status"] = "No candidate or returned-mean physical utility measured; no E1 result."
        persist()
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                      "query_seconds": summary["query_seconds"], "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
                      "output": str(out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        protocol = json.loads(args.config.read_text())
        print(json.dumps(verify(ROOT / protocol["out_dir"])))
    else:
        bootstrap()
        run(args.config)
