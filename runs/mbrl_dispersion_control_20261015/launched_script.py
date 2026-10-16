"""Frozen CartPole dispersion control; parent supervises a CPU-only worker.

The only numerical intervention is a per-instance wrapper around native CEM's
dispersion update. Native optimize, mean updates, legal-bound constraints,
sampling, ModelEnv evaluation and best-solution warm starts remain upstream.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/mbrl_dispersion_control_20261015.json"


class HistoricalParityError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def historical(mode, seed):
    version = 0 if seed < 812 else 1
    base = ROOT / f"runs/mbrl_covariance_pinching_eval_{mode}_metrics_v{version}"
    data = read_json(base / f"episode_{seed}.json")
    plan = read_json(base / "plan.json")["arguments"]
    require(plan["condition"] == "native" and not plan["return_mean_elites"], "Wrong historical CEM")
    require(plan["model"] == f"runs/mbrl_covariance_pinching_{mode}_v0", "Wrong historical checkpoint")
    data["source"] = str((base / f"episode_{seed}.json").relative_to(ROOT))
    return data


def worker(config, output):
    start = time.monotonic()
    summary = {"status": "running", "pairs": [], "active": None,
               "activity": {"checkpoint_loads": 0, "completed_real_episodes": 0,
                            "training_updates": 0, "api_calls": 0, "gpu_model_calls": 0},
               "scope": config["scope"], "complete_panel": False}

    def checkpoint():
        summary["elapsed_seconds"] = time.monotonic() - start
        save_json(output / "summary.json", summary)

    def deadline():
        if time.monotonic() - start >= config["limits"]["hard_wallclock_seconds"]:
            raise TimeoutError("Fixed600-second worker budget exhausted")

    checkpoint()
    try:
        versions = {n: importlib.metadata.version(n) for n in config["runtime"]["required_packages"]}
        require(versions == config["runtime"]["required_packages"], "Runtime package metadata changed")
        require(".".join(map(str, sys.version_info[:3])) == config["runtime"]["python_version"], "Wrong Python")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/mbrl-lib")]
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from mbrl.models import ModelEnv
        from mbrl.planning import TrajectoryOptimizerAgent
        from pilot_mbrl_rank_general import make_model, task_components, reset, FullBankMean
        from audit_cem_dispersion_action_source import source_contract

        summary["source_contract"] = source_contract()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        summary["runtime"] = {"python": sys.version, "executable": sys.executable,
                              "packages": versions, "torch_actual_version": torch.__version__,
                              "device": "cpu", "torch_threads": torch.get_num_threads()}
        env_class, reward_fn, term_fn = task_components("cartpole")

        class QueryAudit:
            def __init__(self, optimizer, model_env, arm):
                self.optimizer, self.model_env, self.arm = optimizer, model_env, arm
                self.original = {name: getattr(optimizer, name) for name in (
                    "_init_population_params", "_sample_population", "_update_population_params")}
                optimizer._init_population_params = self.init
                optimizer._sample_population = self.sample
                optimizer._update_population_params = self.update

            def begin(self, step, obs, warm):
                self.step, self.obs = step, np.asarray(obs).copy()
                self.warm = warm.clone()
                self.global_before = torch.get_rng_state().clone()
                self.model_before = self.model_env._rng.get_state().clone()
                self.iterations, self.stats = [], []
                self.best_value, self.best_iteration, self.best_index, self.best_sequence = -math.inf, None, None, None
                self.initial_dispersion = None

            def init(self, x0):
                mu, dispersion = self.original["_init_population_params"](x0)
                self.initial_dispersion = dispersion.clone()
                require(torch.equal(dispersion, torch.full_like(dispersion, 0.25)), "Changed native initial variance")
                return mu, dispersion

            def sample(self, mu, dispersion, population):
                lb, ub = self.optimizer.lower_bound, self.optimizer.upper_bound
                bound = torch.minimum(((mu-lb)/2).square(), ((ub-mu)/2).square())
                effective = torch.minimum(bound, dispersion)
                record = {"mu": mu.clone(), "dispersion": dispersion.clone(),
                          "bound_variance": bound.clone(), "effective_variance": effective.clone()}
                self.iterations.append(record)
                return self.original["_sample_population"](mu, dispersion, population)

            def update(self, elite, mu, dispersion):
                new_mu, native_dispersion = self.original["_update_population_params"](elite, mu, dispersion)
                returned = (self.initial_dispersion.clone() if self.arm == "fixed_dispersion" else native_dispersion)
                self.iterations[-1].update(native_updated_mu=new_mu.clone(),
                    native_updated_dispersion=native_dispersion.clone(), returned_dispersion=returned.clone())
                return new_mu, returned

            def callback(self, population, values, iteration):
                deadline()
                require(len(self.iterations) == iteration+1, "Unexpected optimizer callback order")
                require(bool(torch.isfinite(values).all()), "Nonfinite native forecast scores")
                # Exact same deterministic diagnostics as the original runner.
                elite_num = 10
                elite_idx = torch.topk(values, elite_num).indices
                elite = population[elite_idx]
                flat, elite_flat = population.reshape(len(population), -1), elite.reshape(len(elite), -1)
                cov, elite_cov = torch.cov(flat.T), torch.cov(elite_flat.T)
                stat = dict(step=self.step, iteration=iteration,
                    population_std_mean=float(flat.std(dim=0).mean()),
                    population_cov_trace=float(torch.trace(cov)),
                    population_cov_maxeig=float(torch.linalg.eigvalsh(cov).max()),
                    elite_std_mean=float(elite_flat.std(dim=0).mean()),
                    elite_cov_trace=float(torch.trace(elite_cov)), value_mean=float(values.mean()),
                    value_max=float(values.max()), value_p90=float(torch.quantile(values, .9)))
                self.stats.append(stat)
                maximum = float(values[elite_idx[0]])
                if maximum > self.best_value:
                    self.best_value, self.best_iteration, self.best_index = maximum, iteration, int(elite_idx[0])
                    self.best_sequence = population[elite_idx[0]].clone()
                self.iterations[-1].update(population=population.clone(), values=values.clone(), elite_indices=elite_idx.clone())

            def finish(self, action, warm_after, path):
                require(len(self.iterations) == 3, "Incomplete native CEM query")
                require(np.array_equal(action, self.best_sequence[0].numpy()), "Actual action not from native strict best candidate")
                expected_warm = self.best_sequence.roll(-1, dims=0)
                expected_warm[-1] = 0.0
                require(torch.equal(warm_after, expected_warm), "Native warm start changed")
                arrays = {name: torch.stack([r[name] for r in self.iterations]).numpy()
                          for name in self.iterations[0]}
                arrays.update(obs=self.obs, warm_before=self.warm.numpy(),
                              warm_after=warm_after.clone().numpy(), returned_sequence=self.best_sequence.numpy(),
                              torch_rng_before=self.global_before.numpy(),
                              torch_rng_after=torch.get_rng_state().numpy(),
                              model_rng_before=self.model_before.numpy(),
                              model_rng_after=self.model_env._rng.get_state().numpy())
                np.savez_compressed(path, **arrays)
                row = {"query_step": self.step, "cache": path.name,
                       "winner_iteration": self.best_iteration, "winner_candidate_index": self.best_index,
                       "best_value": self.best_value, "first_value_max": self.stats[0]["value_max"],
                       "first_round_ceiling": self.stats[0]["value_max"] == 15.0,
                       "iteration_stats": self.stats,
                       "fixed_vs_current_elites_native_update_changes": sum(
                           not torch.equal(r["native_updated_dispersion"], r["returned_dispersion"])
                           for r in self.iterations),
                       "same_elites_native_update_is_local_formula_only": True}
                return row, arrays

            def restore(self):
                for name, method in self.original.items():
                    setattr(self.optimizer, name, method)

        def run_arm(model, mode, seed, arm, reference, native_result, python_rng):
            arm_out = output / f"{mode}_{seed}" / arm
            arm_out.mkdir(parents=True, exist_ok=False)
            result = {"mode": mode, "seed": seed, "arm": arm, "status": "running",
                      "real_steps": 0, "return": None, "history_gate_pass": None,
                      "history_source": reference["source"], "historical_stats_exact": True,
                      "historical_stats_allclose": True, "query_count": 0,
                      "shared_query_rng_checks": 0, "same_input_query_comparisons": 0,
                      "same_input_effective_variance_changed_queries": 0,
                      "same_input_non_ceiling_effective_variance_changed_queries": 0,
                      "same_input_changed_action_queries": 0}
            trajectory, queries = [], []
            env, audit, sampler = None, None, None
            began = time.monotonic()

            def progress():
                result["elapsed_seconds"] = time.monotonic() - began
                if sampler is not None:
                    result["evaluations"] = sampler.calls
                    result["predictive_transitions"] = sampler.predictive_transitions
                save_json(arm_out / "summary.json", result)
                summary["active"] = {"mode": mode, "seed": seed, "arm": arm,
                                     "path": str(arm_out.relative_to(ROOT)),
                                     "completed_real_steps": len(trajectory), "query_started": len(queries)}
                checkpoint()

            try:
                deadline()
                random.setstate(python_rng)
                torch.manual_seed(seed)
                env = env_class()
                env.action_space.seed(seed+50000)
                obs, _ = reset(env, "cartpole", seed)
                if not np.array_equal(obs, reference["trajectory"][0]["obs"]):
                    raise HistoricalParityError("Initial physical observation differs")
                generator = torch.Generator().manual_seed(seed+10000)
                model_env = ModelEnv(env, model, term_fn, reward_fn, generator=generator)
                cfg = OmegaConf.create(dict(_target_="mbrl.planning.CEMOptimizer", num_iterations=3,
                    elite_ratio=.1, population_size=100, alpha=.1, device="cpu",
                    return_mean_elites=False, clipped_normal=False))
                agent = TrajectoryOptimizerAgent(cfg, env.action_space.low.tolist(), env.action_space.high.tolist(),
                                                 planning_horizon=15, replan_freq=1)
                sampler = FullBankMean(model_env, 10, 1)
                agent.set_trajectory_eval_fn(sampler.evaluate)
                audit = QueryAudit(agent.optimizer.optimizer, model_env, arm)
                result["initial_observation"] = obs.tolist()
                old_stats = {(s["step"], s["iteration"]): s for s in reference["summary"]["cem_stats"]}
                progress()
                with (arm_out / "trajectory.jsonl").open("x") as trace, (arm_out / "queries.jsonl").open("x") as query_log:
                    for step in range(200):
                        deadline()
                        audit.begin(step, obs, agent.optimizer.previous_solution)

                        def callback(population, values, iteration):
                            audit.callback(population, values, iteration)
                            if arm == "native":
                                expected = old_stats.get((step, iteration))
                                if expected is None:
                                    raise HistoricalParityError("Native query exceeds historical trajectory")
                                actual = audit.stats[-1]
                                exact = actual == expected
                                allclose = all((v == expected[k] if k in ("step", "iteration", "value_max") else
                                    math.isclose(v, expected[k], abs_tol=config["historical_gate"]["diagnostic_stats_atol"],
                                                 rel_tol=config["historical_gate"]["diagnostic_stats_rtol"]))
                                    for k, v in actual.items())
                                result["historical_stats_exact"] &= exact
                                result["historical_stats_allclose"] &= allclose
                                if not allclose:
                                    result["historical_stat_mismatch"] = {"actual": actual, "expected": expected}
                                    raise HistoricalParityError("Native score maximum or diagnostics exceed registered tolerance")

                        action = agent.act(obs, optimizer_callback=callback)
                        query, arrays = audit.finish(action, agent.optimizer.previous_solution,
                                                     arm_out / f"query_{step:03d}.npz")
                        if arm == "fixed_dispersion" and step < native_result["real_steps"]:
                            native_cache = output / f"{mode}_{seed}" / "native" / f"query_{step:03d}.npz"
                            with np.load(native_cache, allow_pickle=False) as native:
                                rng_checks = {key: np.array_equal(arrays[key], native[key]) for key in (
                                    "torch_rng_before", "torch_rng_after", "model_rng_before", "model_rng_after")}
                                require(all(rng_checks.values()), "Paired RNG consumption differs at shared query index")
                                result["shared_query_rng_checks"] += 1
                                same_input = all(np.array_equal(arrays[key], native[key]) for key in ("obs", "warm_before"))
                                query["comparison_to_native_same_query_index"] = {"rng": rng_checks,
                                    "same_physical_observation_and_warmstart": same_input,
                                    "scope": "Only matching input queries support same-input contrasts; other rows are distinct closed-loop states."}
                                if same_input:
                                    result["same_input_query_comparisons"] += 1
                                    require(np.array_equal(arrays["population"][0], native["population"][0])
                                            and np.array_equal(arrays["values"][0], native["values"][0]),
                                            "First candidate population/scores differ despite matched inputs and RNG")
                                    variance_changed = not np.array_equal(arrays["effective_variance"], native["effective_variance"])
                                    non_ceiling = float(native["values"][0].max()) < 15.0
                                    changed_action = not np.array_equal(action, native["returned_sequence"][0])
                                    query["comparison_to_native_same_query_index"].update(
                                        first_population_and_scores_exact=True, effective_variance_changed=variance_changed,
                                        native_first_round_non_ceiling=non_ceiling, executed_action_changed=changed_action,
                                        full_forecast_scores_changed=not np.array_equal(arrays["values"], native["values"]))
                                    result["same_input_effective_variance_changed_queries"] += int(variance_changed)
                                    result["same_input_non_ceiling_effective_variance_changed_queries"] += int(variance_changed and non_ceiling)
                                    result["same_input_changed_action_queries"] += int(changed_action)
                                    if not non_ceiling:
                                        require(not changed_action and np.array_equal(arrays["returned_sequence"], native["returned_sequence"]),
                                                "Ceiling-locked same-input query changed its best solution")
                        queries.append(query)
                        query_log.write(json.dumps(query) + "\n")
                        query_log.flush()
                        result["query_count"] = len(queries)
                        if arm == "native" and not np.array_equal(action, reference["trajectory"][step]["action"]):
                            raise HistoricalParityError("Native executed-action proposal differs from historical action")
                        nxt, reward, done, truncated, _ = env.step(action)
                        row = dict(step=step, obs=obs.tolist(), action=action.tolist(), next_obs=nxt.tolist(),
                                   reward=float(reward), terminated=bool(done), truncated=bool(truncated))
                        trajectory.append(row)
                        trace.write(json.dumps(row) + "\n")
                        trace.flush()
                        result.update(real_steps=len(trajectory), **{"return": math.fsum(r["reward"] for r in trajectory)})
                        if arm == "native" and row != reference["trajectory"][step]:
                            result["historical_trajectory_mismatch"] = {"actual": row, "expected": reference["trajectory"][step]}
                            raise HistoricalParityError("Native physical/reward trajectory differs from historical row")
                        obs = nxt
                        if step % 25 == 0 or done or truncated:
                            progress()
                            print(json.dumps({"mode": mode, "seed": seed, "arm": arm, "steps": len(trajectory),
                                              "return": result["return"]}), flush=True)
                        if done or truncated:
                            break
                if arm == "native":
                    if trajectory != reference["trajectory"] or result["return"] != reference["summary"]["return_"]:
                        raise HistoricalParityError("Native complete trajectory/return/length mismatch")
                    result["history_gate_pass"] = True
                else:
                    result["history_gate_pass"] = "paired_native_passed_before_this_arm"
                    if mode == "interior" and seed == 813:
                        require(len(trajectory) == 200 and trajectory == reference["trajectory"],
                                "Prespecified all-ceiling negative-control trajectory changed")
                        result["prespecified_all_ceiling_negative_control_exact"] = True
                result["status"] = "completed_episode"
                result["reached_episode_cap"] = len(trajectory) == 200 and not trajectory[-1]["terminated"]
                summary["activity"]["completed_real_episodes"] += 1
            except HistoricalParityError as exc:
                result.update(status="historical_parity_failed_fixed_arm_blocked", history_gate_pass=False,
                              exception_type=type(exc).__name__, exception=str(exc))
            except BaseException as exc:
                result.update(status="partial_resource_deferred" if isinstance(exc, TimeoutError) else "technical_error",
                              exception_type=type(exc).__name__, exception=str(exc))
                raise
            finally:
                if audit is not None:
                    audit.restore()
                if env is not None:
                    env.close()
                result["real_steps"] = len(trajectory)
                result["return_scope"] = "Complete episode" if result["status"] == "completed_episode" else "Partial observed reward sum only; not a completed task outcome"
                progress()
            return result

        for spec in config["models"]:
            deadline()
            mode = spec["mode"]
            model = make_model("cartpole")
            model.load(ROOT / spec["path"])
            model.eval()
            summary["activity"]["checkpoint_loads"] += 1
            require(all(p.device.type == "cpu" for p in model.parameters()), "Non-CPU parameter")
            frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
            normalizer = model.input_normalizer
            frozen_norm = {k: getattr(normalizer, k).clone() for k in ("mean", "std")}
            try:
                for seed in config["seeds"]:
                    reference = historical(mode, seed)
                    python_rng = random.getstate()
                    pair = {"mode": mode, "seed": seed, "status": "running", "native": None, "fixed": None}
                    summary["pairs"].append(pair)
                    pair["native"] = run_arm(model, mode, seed, "native", reference, None, python_rng)
                    if pair["native"]["history_gate_pass"] is not True:
                        pair["status"] = "historical_parity_failed_fixed_not_run"
                        checkpoint()
                        continue
                    pair["fixed"] = run_arm(model, mode, seed, "fixed_dispersion", reference, pair["native"], python_rng)
                    pair["delta_return"] = pair["fixed"]["return"] - pair["native"]["return"]
                    pair["status"] = "completed_pair"
                    checkpoint()
            finally:
                weights_equal = all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
                norm_equal = all(torch.equal(v, getattr(normalizer, k)) for k, v in frozen_norm.items())
                summary.setdefault("frozen_model_checks", []).append({"mode": mode, "state_dict_exact": weights_equal,
                    "input_normalizer_mean_std_exact": norm_equal, "parameter_tensors": len(frozen)})
                require(weights_equal and norm_equal, "Protected model or normalizer changed")
                del model
        summary["complete_panel"] = len(summary["pairs"]) == 12 and all(p["status"] == "completed_pair" for p in summary["pairs"])
        if summary["complete_panel"]:
            interactions = []
            for seed in config["seeds"]:
                rows = {p["mode"]: p for p in summary["pairs"] if p["seed"] == seed}
                delta = {m: rows[m]["delta_return"] for m in ("random", "boundary", "interior")}
                interactions.append({"seed": seed, "delta_return": delta,
                    "boundary_minus_random_delta": delta["boundary"] - delta["random"],
                    "boundary_minus_interior_delta": delta["boundary"] - delta["interior"]})
            summary["descriptive_interactions_not_significance"] = interactions
        summary["status"] = "completed_fixed_panel" if summary["complete_panel"] else "completed_panel_with_historical_gates_failed"
        summary["interpretation_rule"] = config["analysis"]
    except BaseException as exc:
        summary.update(status="partial_resource_deferred" if isinstance(exc, TimeoutError) else "technical_error",
                       exception_type=type(exc).__name__, exception=str(exc))
    finally:
        summary["last_active"] = summary.get("active")
        summary["active"] = None
        checkpoint()
        print(json.dumps({k: summary[k] for k in ("status", "complete_panel", "activity", "elapsed_seconds")}), flush=True)
    return 0 if summary["status"].startswith("completed_") else 1


def validate_config(config):
    require(config["status"] == "approved", "Root must review/freeze the protocol before execution")
    require(config["models"] == [{"mode": m, "path": f"runs/mbrl_covariance_pinching_{m}_v0"}
                                for m in ("random", "boundary", "interior")], "Changed model panel")
    require(config["seeds"] == [810, 811, 812, 813], "Changed seed panel")
    require(config["planner"] == {"population": 100, "iterations": 3, "horizon": 15, "particles": 10,
        "elite_ratio": .1, "alpha": .1, "return_mean_elites": False, "clipped_normal": False, "max_real_steps": 200},
        "Changed native planner budget")
    require(config["limits"]["hard_wallclock_seconds"] == 600, "Changed total deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
    output = (args.output or ROOT / config["default_output"]).resolve()
    if args.worker:
        return worker(config, output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", config)
    (output / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    source_dir = output / "source_snapshots"
    source_dir.mkdir()
    for path in ("scripts/pilot_mbrl_rank_general.py", "scripts/pilot_mbrl_coupling.py",
                 "vendor/mbrl-lib/mbrl/planning/trajectory_opt.py", "vendor/mbrl-lib/mbrl/models/model_env.py",
                 "vendor/mbrl-lib/mbrl/models/gaussian_mlp.py", "scripts/audit_cem_dispersion_action_source.py"):
        (source_dir / path.replace("/", "__")).write_bytes((ROOT / path).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(args.config.resolve()),
               "--output", str(output)]
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (output / "worker.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=config["limits"]["hard_wallclock_seconds"])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            path = output / "summary.json"
            result = read_json(path) if path.exists() else {}
            result.update(status="partial_resource_deferred", parent_hard_timeout=True, complete_panel=False,
                          partial_scope="Last saved summaries and flushed per-step/query files only; no unfinished episode is a task failure.")
            result["last_active"] = result.get("active")
            result["active"] = None
            save_json(path, result)
            code = 1
    print(json.dumps({"worker_exit_code": code, "output": str(output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
