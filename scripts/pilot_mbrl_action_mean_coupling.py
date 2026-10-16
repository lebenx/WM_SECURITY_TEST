"""Fixed CPU native-CEM test of marginal-preserving action-mean coupling.

Only new outputs are written. This runner never trains or changes upstream code.
The sampler exposes scalar N-particle forecast returns, not particle transcripts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/mbrl-lib")]

import numpy as np
import torch
from omegaconf import OmegaConf
from mbrl.env import reward_fns, termination_fns
from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.models import GaussianMLP, OneDTransitionRewardModel, ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from wm_security.rank_coupling import optimize_rank_phases


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def action_means(values, actions, elite_num):
    """Use exactly native topk; batched diagnostic only, no planner mutation."""
    indices = values.topk(elite_num, dim=-1).indices
    return actions[:, 0, 0][indices].mean(-1)


def optimize_action_phases(bank, actions, elite_num, sweeps, tolerance):
    ordered = bank.sort(dim=0).values
    k, population = bank.shape
    rows = torch.arange(k)
    phases = torch.zeros(population, dtype=torch.long)
    current = ordered.clone()
    before = best = float(action_means(current, actions, elite_num).mean())
    evaluations = 1
    for _ in range(sweeps):
        changed = False
        for column in range(population):
            alternatives = current.expand(k, k, population).clone()
            alternatives[:, :, column] = ordered[(rows[None, :] + rows[:, None]) % k, column]
            objectives = action_means(alternatives, actions, elite_num).mean(-1)
            evaluations += k
            choice = int(objectives.argmax())
            candidate = float(objectives[choice])
            if candidate > best + tolerance:
                phases[column] = choice
                current = alternatives[choice].clone()
                best = candidate
                changed = True
        if not changed:
            break
    return phases, {"before": before, "after": best, "phase_candidates": evaluations}


def verify_permutations(bank, phases):
    k, population = bank.shape
    ordered, original_indices = bank.sort(dim=0)
    indices = (torch.arange(k)[:, None] + phases[None, :]) % k
    selected = ordered.gather(0, indices)
    original = original_indices.gather(0, indices)
    expected = torch.arange(k)[:, None].expand(k, population)
    scalar_match = torch.equal(selected.sort(dim=0).values, ordered)
    bijection = torch.equal(original.sort(dim=0).values, expected)
    assert scalar_match and bijection
    return selected, {"scalar_column_multisets_exact": scalar_match,
                      "original_bank_indices_bijective": bijection,
                      "columns": population, "rows_per_column": k}


class Sampler:
    def __init__(self, model_env, cfg, arm, out):
        self.env, self.cfg, self.arm, self.out = model_env, cfg, arm, out
        self.selection_rng = torch.Generator().manual_seed(cfg["seed"] + 30000)
        self.row_rng = torch.Generator().manual_seed(cfg["seed"] + 40000)
        self.calls = self.transitions = self.phase_candidates = 0
        self.model_seconds = self.phase_seconds = 0.0
        self.step = 0
        self.records = []
        self.first_queries = []

    def evaluate(self, initial_state, actions):
        cfg = self.cfg
        iteration = self.calls % cfg["iterations"]
        started = time.monotonic()
        bank = torch.stack([self.env.evaluate_action_sequences(actions, initial_state, cfg["particles"])
                            for _ in range(cfg["bank_size"])])
        self.model_seconds += time.monotonic() - started
        assert torch.isfinite(bank).all()
        assert tuple(actions.shape) == (cfg["population"], cfg["horizon"], 1)
        ordered = bank.sort(dim=0).values
        # Consume the same independent selection stream on EVERY query in both arms.
        indices = torch.randint(cfg["bank_size"], (len(actions),), generator=self.selection_rng)
        independent_indices = indices.clone()
        phases = torch.zeros(len(actions), dtype=torch.long)
        info = None
        row = None
        if self.arm == "positive_action_mean" and iteration == cfg["iterations"] - 1:
            started = time.monotonic()
            phases, info = optimize_action_phases(bank, actions, self.elite_num,
                                                  cfg["phase_sweeps"], cfg["phase_improvement_tolerance"])
            self.phase_seconds += time.monotonic() - started
            self.phase_candidates += info["phase_candidates"]
            _, proof = verify_permutations(bank, phases)
            # No random row is available to the phase optimizer.
            row = int(torch.randint(cfg["bank_size"], (), generator=self.row_rng))
            indices = (row + phases) % cfg["bank_size"]
            info.update(proof)
        scores = ordered[indices, torch.arange(len(actions))].clone()
        self.calls += 1
        self.transitions += cfg["bank_size"] * len(actions) * cfg["horizon"] * cfg["particles"]
        self.last = {"bank": bank.clone(), "actions": actions.clone(), "scores": scores.clone(),
                     "independent_indices": independent_indices, "indices": indices.clone(),
                     "phases": phases.clone(), "iteration": iteration, "row": row, "info": info}
        if self.step == 0:
            self.first_queries.append({key: tensor_digest(value) for key, value in self.last.items()
                                       if isinstance(value, torch.Tensor)})
            np.savez_compressed(self.out / f"first_query_{iteration}.npz",
                                **{key: value.numpy() for key, value in self.last.items()
                                   if isinstance(value, torch.Tensor)})
        if iteration == cfg["iterations"] - 1:
            np.savez_compressed(self.out / f"final_query_{self.step:03d}.npz",
                                **{key: value.numpy() for key, value in self.last.items()
                                   if isinstance(value, torch.Tensor)})
        return scores

    @property
    def elite_num(self):
        return int(np.ceil(self.cfg["population"] * self.cfg["elite_ratio"]))


def first_plan_audit(last, mu_before, cfg, out, plan_seconds):
    audit_started = time.monotonic()
    bank, actions = last["bank"], last["actions"]
    elite = int(np.ceil(cfg["population"] * cfg["elite_ratio"]))
    started = time.monotonic()
    phases, info = optimize_action_phases(bank, actions, elite, cfg["phase_sweeps"],
                                         cfg["phase_improvement_tolerance"])
    phase_seconds = time.monotonic() - started
    shifted, proof = verify_permutations(bank, phases)
    ordered = bank.sort(dim=0).values
    old_started = time.monotonic()
    old_phases, old_info = optimize_rank_phases(bank, elite, sweeps=2)
    old_diagnostic_seconds = time.monotonic() - old_started
    old_shifted, _ = verify_permutations(bank, old_phases)
    rng = torch.Generator().manual_seed(cfg["seed"] + 60000)
    independent_rows = torch.randint(cfg["bank_size"], (1024, cfg["population"]), generator=rng)
    independent_values = ordered.gather(0, independent_rows)
    alpha = cfg["alpha"]
    def returned(values):
        return alpha * mu_before[0, 0] + (1 - alpha) * action_means(values, actions, elite)
    diagnostics = {}
    for name, values in [("shared", ordered), ("positive_action_mean", shifted),
                         ("old_rank_targeted", old_shifted), ("bank_mean", bank.mean(0)[None]),
                         ("independent_mc_1024", independent_values)]:
        acts = returned(values)
        diagnostics[name] = {"expected_or_mc_mean_returned_first_action": float(acts.mean()),
                             "min": float(acts.min()), "max": float(acts.max()),
                             "row_count": len(acts), "exact_row_enumeration": name != "independent_mc_1024"}
    gain = diagnostics["positive_action_mean"]["expected_or_mc_mean_returned_first_action"] - diagnostics["shared"]["expected_or_mc_mean_returned_first_action"]
    audit_seconds = time.monotonic() - audit_started
    predicted_seconds = 2 * cfg["max_steps"] * plan_seconds + cfg["max_steps"] * phase_seconds + audit_seconds
    result = {"status": "completed_operation_audit", **proof, "phase_objective": info,
              "old_targeted_objective": old_info, "mu_before_final": mu_before.tolist(),
              "diagnostics": diagnostics, "signed_expected_action_gain_vs_shared": gain,
              "phase_seconds": phase_seconds, "first_native_plan_seconds": plan_seconds,
              "total_offline_operation_audit_seconds": audit_seconds,
              "old_targeted_diagnostic_search_seconds": old_diagnostic_seconds,
              "pair_runtime_extrapolation_seconds": predicted_seconds,
              "runtime_extrapolation_not_measured_episodes": True,
              "operation_signal_gate_pass": gain > 1e-6,
              "runtime_gate_pass": predicted_seconds <= cfg["runtime_limit_seconds"],
              "full_trajectory_marginal_empirically_checked": False,
              "diagnostic_controls_have_no_episode_utility": True}
    np.savez_compressed(out / "operation_audit.npz", phases=phases.numpy(),
                        old_phases=old_phases.numpy(), mu_before=mu_before.numpy())
    save(out / "operation_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mbrl_action_mean_coupling_20261015.json")
    parser.add_argument("--out", default="runs/mbrl_action_mean_coupling_20261015")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["seed"])
    model = OneDTransitionRewardModel(
        GaussianMLP(5, 4, "cpu", num_layers=2, ensemble_size=5, hid_size=64,
                    deterministic=False, propagation_method="fixed_model",
                    activation_fn_cfg={"_target_": "torch.nn.LeakyReLU", "negative_slope": .01}),
        target_is_delta=True, normalize=True, learned_rewards=False)
    model.load(ROOT / cfg["model"])
    model.eval()
    source_paths = [cfg_path, Path(__file__), ROOT / "src/wm_security/rank_coupling.py",
                    ROOT / "vendor/mbrl-lib/mbrl/planning/trajectory_opt.py"]
    source_paths += [ROOT / cfg["model"] / name for name in ["model.pth", "env_stats.pickle"]]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    frozen_parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
    summary = {"status": "running", "configuration": cfg, "source_hashes": sources,
               "runtime": {"python": sys.version, "executable": sys.executable,
                           "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1},
               "rows": [], "api_requests": 0, "training_updates": 0, "gpu_work": False}
    save(out / "summary.json", summary)
    try:
        for arm in cfg["arms"]:
            arm_out = out / arm
            arm_out.mkdir()
            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            env = CartPoleEnv()
            env.action_space.seed(cfg["seed"] + 50000)
            obs, _ = env.reset(cfg["seed"])
            summary.setdefault("initial_observations", {})[arm] = obs.tolist()
            if arm != "independent":
                assert np.array_equal(obs, np.asarray(summary["initial_observations"]["independent"]))
                summary["paired_initial_observation_exact"] = True
            model_env = ModelEnv(env, model, termination_fns.cartpole, reward_fns.cartpole,
                                 generator=torch.Generator().manual_seed(cfg["seed"] + 10000))
            native_cfg = OmegaConf.create({"_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": cfg["iterations"], "elite_ratio": cfg["elite_ratio"],
                "population_size": cfg["population"], "alpha": cfg["alpha"], "device": "cpu",
                "return_mean_elites": cfg["return_mean_elites"], "clipped_normal": cfg["clipped_normal"]})
            agent = TrajectoryOptimizerAgent(native_cfg, env.action_space.low.tolist(),
                      env.action_space.high.tolist(), planning_horizon=cfg["horizon"], replan_freq=1)
            sampler = Sampler(model_env, cfg, arm, arm_out)
            agent.set_trajectory_eval_fn(sampler.evaluate)
            trajectory, audit_records = [], []
            episode_started = time.monotonic()
            try:
                for step in range(cfg["max_steps"]):
                    if time.monotonic() - started > cfg["runtime_limit_seconds"]:
                        raise TimeoutError("Fixed 1800-second run budget reached; no extra episode or retuning.")
                    sampler.step = step
                    reconstructed_mu = agent.optimizer.previous_solution.clone()
                    final_mu_before = None
                    def callback(population, values, iteration):
                        nonlocal reconstructed_mu, final_mu_before
                        assert torch.isfinite(values).all()
                        if iteration == cfg["iterations"] - 1:
                            final_mu_before = reconstructed_mu.clone()
                        elite = population[values.topk(sampler.elite_num).indices]
                        reconstructed_mu = cfg["alpha"] * reconstructed_mu + (1 - cfg["alpha"]) * elite.mean(0)
                    plan_started = time.monotonic()
                    action = agent.act(obs, optimizer_callback=callback)
                    plan_seconds = time.monotonic() - plan_started
                    assert np.array_equal(action, reconstructed_mu[0].numpy())
                    if step == 0 and arm == "independent":
                        summary["operation_audit"] = first_plan_audit(sampler.last, final_mu_before,
                                                                       cfg, out, plan_seconds)
                        summary["first_clean_queries"] = sampler.first_queries
                        audit = summary["operation_audit"]
                        print(json.dumps({"event": "operation_audit", **audit}), flush=True)
                        if not audit["operation_signal_gate_pass"] or not audit["runtime_gate_pass"]:
                            summary["status"] = ("stopped_no_operation_signal" if not audit["operation_signal_gate_pass"]
                                                 else "resource_blocked_before_episode")
                            summary["real_environment_steps"] = 0
                            save(out / "summary.json", summary)
                            return
                    if step == 0 and arm == "positive_action_mean":
                        previous = summary["first_clean_queries"]
                        current = sampler.first_queries
                        checks = {"first_four_full_query_records_equal": current[:4] == previous[:4],
                                  "final_actions_exact": current[4]["actions"] == previous[4]["actions"],
                                  "final_bank_exact": current[4]["bank"] == previous[4]["bank"],
                                  "all_independent_row_vectors_exact": all(c["independent_indices"] == p["independent_indices"]
                                                                         for c, p in zip(current, previous))}
                        assert all(checks.values()), checks
                        summary["first_plan_paired_checks"] = checks
                    nxt, reward, done, truncated, _ = env.step(action)
                    trajectory.append({"step": step, "obs": obs.tolist(), "action": action.tolist(),
                        "next_obs": nxt.tolist(), "reward": float(reward), "terminated": bool(done),
                        "truncated": bool(truncated), "plan_seconds": plan_seconds})
                    record = {"step": step, "mu_before_final_first": float(final_mu_before[0, 0]),
                              "returned_action": float(action[0]), "native_mean_exact": True,
                              "fresh_shared_row": sampler.last["row"], "phase_info": sampler.last["info"]}
                    audit_records.append(record)
                    with (arm_out / "trajectory.jsonl").open("a") as stream:
                        stream.write(json.dumps(trajectory[-1], allow_nan=False) + "\n")
                    with (arm_out / "operation_records.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    obs = nxt
                    if step == 0 or (step + 1) % 10 == 0 or done or truncated:
                        print(json.dumps({"arm": arm, "steps": step + 1,
                              "real_return": sum(row["reward"] for row in trajectory),
                              "elapsed_seconds": time.monotonic() - episode_started}), flush=True)
                    if done or truncated:
                        break
                row = {"arm": arm, "seed": cfg["seed"], "real_return": sum(r["reward"] for r in trajectory),
                       "steps": len(trajectory), "terminated": trajectory[-1]["terminated"],
                       "truncated": trajectory[-1]["truncated"],
                       "episode_cap_reached": len(trajectory) == cfg["max_steps"],
                       "seconds": time.monotonic() - episode_started,
                       "native_queries": sampler.calls, "predictive_transitions": sampler.transitions,
                       "phase_candidates": sampler.phase_candidates, "model_seconds": sampler.model_seconds,
                       "phase_seconds": sampler.phase_seconds, "native_mean_exact_all_steps": True}
                save(arm_out / "episode.json", {"summary": row, "trajectory": trajectory,
                                               "operation_records": audit_records})
                summary["rows"].append(row)
                save(out / "summary.json", summary)
                if arm == "independent" and (row["real_return"] != cfg["max_steps"] or row["terminated"]):
                    summary.update(status="stopped_inadequate_clean", scientific_result="intervention_not_run",
                                   real_environment_steps=row["steps"], elapsed_seconds=time.monotonic() - started)
                    save(out / "summary.json", summary)
                    return
            finally:
                env.close()
        assert all(torch.equal(frozen_parameters[key], value) for key, value in model.state_dict().items())
        assert all(digest(ROOT / path) == value for path, value in sources.items())
        clean, attack = summary["rows"]
        summary.update(status="completed", model_parameters_unchanged=True,
                       real_environment_steps=clean["steps"] + attack["steps"],
                       attack_minus_clean_return=attack["real_return"] - clean["real_return"],
                       scientific_result=("conditional_local_harm" if attack["real_return"] < clean["real_return"]
                                          else "no_observed_harm_for_fixed_operation"),
                       elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        if "sampler" in locals() and "trajectory" in locals():
            summary["partial_arm"] = {"arm": arm, "real_steps": len(trajectory),
                 "real_return_so_far": sum(row["reward"] for row in trajectory),
                 "native_queries": sampler.calls, "predictive_transitions": sampler.transitions,
                 "phase_candidates": sampler.phase_candidates, "model_seconds": sampler.model_seconds,
                 "phase_seconds": sampler.phase_seconds, "is_completed_episode": False}
        summary.update(status="resource_incomplete" if isinstance(error, TimeoutError) else "incomplete_error",
                       error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic() - started)
        save(out / "summary.json", summary)
        raise
    save(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ["status", "rows", "attack_minus_clean_return",
                                                  "scientific_result", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
