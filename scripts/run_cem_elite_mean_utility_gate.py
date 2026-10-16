#!/usr/bin/env python3
"""R003: one native CEM query and11 actual15-control branches, no attacker."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/cem_elite_mean_utility_gate_v1"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    from capture_decision_leverage_native_cem import gpu_info, process_gpu_mib
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started, env, torch = time.monotonic(), None, None
    report = {"status": "starting", "registration": "IDEAS.md:R003", "pid": os.getpid(),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "physical_seed": 3000,
              "scope": "One root, unchanged native CEM; actual mean/elite prefix utility. No attacker, external inducer, attack-success or population claim.",
              "native": {"samples": 300, "iterations": 15, "elites": 10, "horizon": 6,
                         "num_act_stepped": 3, "controls_per_macro": 5, "local_seed": 1},
              "primary_rule": "mean terminal reward at15 < minimum of all10 final elites terminal reward at15 minus1e-6",
              "runtime": {"wall_limit_seconds": 300, "allocator_cap_gib": 7, "sampled_process_budget_mib": 8192,
                          "process_limit_scope": "Sampled between cost calls/replays; allocator cap is separately enforced. No continuous total-process memory maximum guarantee.",
                          "microbatch": False, "retries": 0},
              "activity": {"native_query_calls": 0, "cost_calls": 0, "predictor_forwards": 0,
                           "expert_setup_controls": 0, "root_warmup_controls": 0,
                           "native_mean_execution_controls": 0, "counterfactual_elite_controls": 0,
                           "extra_encode_or_forecast_calls": 0, "api_calls": 0, "network_attempts": 0, "training_updates": 0},
              "checkpoint_loads": {}, "branches": [], "checks": {}, "sampled_process_peak_mib": 0,
              "primary_condition": None}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        write(OUT / "summary.json", report)

    def expired(*_):
        raise TimeoutError("Registered300-second wall limit; no retry or protocol reduction")

    def no_network(*_, **__):
        report["activity"]["network_attempts"] += 1
        raise RuntimeError("Network forbidden")

    def memory_check():
        memory = process_gpu_mib()
        report["sampled_process_peak_mib"] = max(report["sampled_process_peak_mib"], memory)
        require(memory <= 8192, "Sampled CUDA process memory exceeded8GiB")

    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, 300)
    persist()
    try:
        report["gpu_at_start"] = gpu_info()
        require(report["gpu_at_start"]["memory.free"] >= 8192, "Insufficient startup GPU memory")
        import numpy as np
        import torch as torch_module
        import mujoco
        from tensordict import TensorDict
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, setup_episode, trace_step, physical, MODEL, DINO_WEIGHTS
        from probe_dino_action_saturation_gate import reconstruct_and_snapshot
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        torch = torch_module
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(7 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        require(mujoco.__version__ == "3.3.0" and str(ROOT / ".deps/dino_mujoco330") in mujoco.__file__, "Wrong MuJoCo runtime")
        report["runtime"].update(torch=torch.__version__, mujoco=mujoco.__version__, renderer="osmesa", threads=4,
                                 interpreter=sys.executable, matmul_tf32=False, cudnn_tf32=True)
        cfg = cfg_for(OUT, "cuda")
        require(cfg.local_seed == 1, "Planner RNG seed changed")
        original_load = torch.load

        def counted_load(path, *a, **kw):
            name = str(Path(path).resolve())
            require(name in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and name not in report["checkpoint_loads"], "Unexpected/repeated checkpoint load")
            report["checkpoint_loads"][name] = 1
            return original_load(path, *a, **kw)

        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            guards.enter_context(patch.object(torch, "load", side_effect=counted_load))
            report["status"] = "loading_model"; persist()
            model, pre, report["model_load"] = load_model(cfg, "cuda")
            require(not model.training and all(not p.requires_grad for p in model.parameters()), "Model not frozen")
            memory_check()
            env, initial, goal, report["setup"] = setup_episode(cfg, 3000, OUT / "setup")
            report["activity"].update(expert_setup_controls=100, root_warmup_controls=1)
            with np.load(OUT / "setup/setup.npz", allow_pickle=False) as f:
                root = {k: f[k].copy() for k in f.files}
            restore, report["restore"] = reconstruct_and_snapshot(env, root, mujoco, torch, np)
            report["restore"]["source"] = "fresh seed3000 snapshot, not historical replay reconstruction"
            report["restore"].pop("historical_source_prefix_verified", None)
            agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
            agent.set_goal(goal)
            planner = agent.planner
            require((planner.num_samples, planner.iterations, planner.num_elites, planner.horizon, planner.num_act_stepped) == (300, 15, 10, 6, 3), "CEM geometry changed")
            require(planner.momentum_mean == planner.momentum_std == 0 and planner.max_norms is None, "CEM mean operator changed")
            original_cost, original_unroll, original_objective = planner.cost_function, planner.unroll, planner.objective
            require(original_objective.alpha == .1 and original_objective.sum_all_diffs is False, "Native objective changed")
            capture = {}

            def cost_observer(actions, z_init):
                costs = original_cost(actions, z_init)
                report["activity"]["cost_calls"] += 1
                if report["activity"]["cost_calls"] == 15:
                    capture.update(actions=actions.detach().clone(), costs=costs.detach().clone())
                memory_check(); persist()
                return costs

            def count_forward(*_):
                report["activity"]["predictor_forwards"] += 1

            hook = model.model.predictor.register_forward_pre_hook(count_forward)
            planner.cost_function = cost_observer
            report["status"] = "native_first_query"; report["activity"]["native_query_calls"] += 1; persist()
            query_start = time.monotonic()
            returned = agent.act(initial, steps_left=max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1))
            torch.cuda.synchronize()
            report["query_seconds"] = time.monotonic() - query_start
            hook.remove(); planner.cost_function = original_cost
            require(report["activity"]["cost_calls"] == 15 and capture["actions"].shape == (6, 300, 20), "Incomplete native query")
            require(planner.unroll == original_unroll and planner.objective is original_objective, "Native consumer replaced")
            elite_ids = torch.topk(-capture["costs"], 10, dim=0).indices
            elite_actions = capture["actions"][:, elite_ids]
            elite_mean = elite_actions.mean(dim=1)
            native_mean = planner._prev_mean
            require(torch.equal(elite_mean, native_mean), "Returned full mean differs from same-GPU final elite mean")
            require(torch.equal(returned.cpu(), native_mean[:3].cpu()), "Returned prefix differs from native mean")
            report["checks"].update(final_mean_exact_to_elite_mean=True, returned_prefix_exact=True,
                                    native_unroll_and_objective_unchanged=True, original_root_restoration=True)
            # Execute native mean first, followed by all10 fixed final elites.
            normalized = torch.cat([native_mean.unsqueeze(1), elite_actions], dim=1).cpu()
            controls = pre.denormalize_actions(normalized[:3].permute(1, 0, 2).reshape(11, 15, 4))
            require(controls.shape == (11, 15, 4) and torch.isfinite(controls).all(), "Invalid physical action prefix")
            require(torch.equal(controls[0], pre.denormalize_actions(returned.cpu().reshape(15, 4))), "Native executed control reshape differs")
            np.savez_compressed(OUT / "native_query.npz", final_actions=capture["actions"].cpu().numpy(),
                                final_costs=capture["costs"].cpu().numpy(), elite_indices=elite_ids.cpu().numpy(),
                                elite_actions=elite_actions.cpu().numpy(), full_mean=native_mean.cpu().numpy(),
                                returned_prefix=returned.cpu().numpy(), physical_prefixes=controls.numpy())
            report["elite_indices"] = elite_ids.cpu().tolist()
            report["status"] = "actual_prefix_replays"; persist()
            for branch in range(11):
                restore()
                name = "native_mean" if branch == 0 else f"elite{branch-1:02d}"
                directory = OUT / name; directory.mkdir()
                rows = []
                with (directory / "trace.jsonl").open("x") as handle:
                    for step in range(15):
                        _, _, row = trace_step(env, controls[branch, step].clone(), 1, step)
                        report["activity"]["native_mean_execution_controls" if branch == 0 else "counterfactual_elite_controls"] += 1
                        require(not row["done"] and not row["truncated"] and not env.proprio_env.unwrapped._did_see_sim_exception, "Replay failed/terminated early")
                        require(np.array_equal(physical(env)["target"], root["target"]), "Original target changed")
                        rows.append(row); handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                utility = {"name": name, "controls": 15, "source_population_index": None if branch == 0 else report["elite_indices"][branch-1],
                           "terminal_reward": rows[-1]["reward"], "cumulative_reward": sum(r["reward"] for r in rows),
                           "terminal_reach_distance": rows[-1]["reach_distance"], "terminal_success": bool(rows[-1]["success"]),
                           "ever_success": any(r["success"] for r in rows),
                           "first_success_control": next((i+1 for i,r in enumerate(rows) if r["success"]), None)}
                report["branches"].append(utility); memory_check(); persist()
                print(json.dumps(utility), flush=True)
            restore()
            mean, elites = report["branches"][0], report["branches"][1:]
            lo, hi = min(e["terminal_reward"] for e in elites), max(e["terminal_reward"] for e in elites)
            report["primary_condition"] = mean["terminal_reward"] < lo - 1e-6
            report["comparison"] = {"mean_terminal_reward": mean["terminal_reward"], "elite_min_terminal_reward": lo,
                                    "elite_max_terminal_reward": hi, "mean_minus_elite_min": mean["terminal_reward"] - lo,
                                    "elites_with_higher_terminal_reward_than_mean": sum(e["terminal_reward"] > mean["terminal_reward"] for e in elites)}
            require(report["activity"]["predictor_forwards"] == 180, "Unexpected native model call count")
            report["status"] = "completed_single_root_native_mean_utility_gate"
            report["decision"] = "necessary_condition_observed_external_inducer_unproved" if report["primary_condition"] else "registered_condition_absent_close_fixed_gate_no_seed_sweep"
    except BaseException as error:
        report.update(status="resource_interrupted" if isinstance(error, TimeoutError) or "out of memory" in str(error).lower() else "failed",
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            env.close()
        if torch is not None and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({"status": report["status"], "primary_condition": report["primary_condition"], "comparison": report.get("comparison"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    from capture_decision_leverage_native_cem import bootstrap
    bootstrap()
    main()
