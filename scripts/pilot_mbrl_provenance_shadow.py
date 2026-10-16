"""Short native gate for counterfactual-provenance shadowing.

The candidate is deliberately narrow and falsifiable: legal boundary actions
are executed in the real Pusher simulator before planning starts.  We then
measure whether ensemble disagreement remains low while model-vs-simulator
rollout error and native CEM return degrade.  Interior and random prefixes are
matched controls; no observation, reward, model weight, or planner code is
modified.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env import reward_fns, termination_fns
from mbrl.env.pets_pusher import PusherEnv
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from pilot_mbrl_rank_general import make_model, model_hash, reset


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def env_state(env):
    return env.data.qpos.copy(), env.data.qvel.copy()


def restore_state(env, state):
    env.set_state(state[0].copy(), state[1].copy())


def prefix_actions(rng, mode, n, dim):
    if mode == "clean":
        return np.zeros((0, dim), dtype=np.float32)
    if mode == "boundary":
        # Valid action-space values at the physical boundary; signs vary so
        # this is not a single hand-tuned trajectory.
        return rng.choice(np.array([-2.0, 2.0], dtype=np.float32), size=(n, dim))
    if mode == "interior":
        return rng.uniform(-0.5, 0.5, size=(n, dim)).astype(np.float32)
    if mode == "random":
        return rng.uniform(-2.0, 2.0, size=(n, dim)).astype(np.float32)
    raise ValueError(mode)


@torch.no_grad()
def one_step_disagreement(model, obs, action_batch):
    obs_batch = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], len(action_batch), axis=0)
    model_in = model._get_model_input(obs_batch, action_batch)
    means, _ = model.model.forward(model_in, use_propagation=False)
    # E x B x D: epistemic variance of the predictive mean, averaged over
    # transitions and state dimensions.  Aleatoric log variance is excluded.
    return float(means.var(dim=0).mean().cpu()), means.mean(dim=0).cpu().numpy()


@torch.no_grad()
def rollout_error(model, env, start_obs, start_state, action_sequences):
    """Compare ensemble-mean imagined states to truthful simulator rollouts."""
    errors, disagreements = [], []
    for actions in action_sequences:
        obs_model = np.asarray(start_obs, dtype=np.float32).copy()
        restore_state(env, start_state)
        truth = np.asarray(start_obs, dtype=np.float32).copy()
        seq_errors, seq_dis = [], []
        for action in actions:
            obs_batch = truth[None, :]
            model_in = model._get_model_input(obs_batch, np.asarray(action, dtype=np.float32)[None, :])
            means, _ = model.model.forward(model_in, use_propagation=False)
            mean_delta = means.mean(dim=0)[0].cpu().numpy()
            obs_model = obs_model + mean_delta
            truth, _, done, truncated, _ = env.step(action)
            truth = np.asarray(truth, dtype=np.float32)
            seq_errors.append(float(np.mean((obs_model - truth) ** 2) ** 0.5))
            seq_dis.append(float(means[:, 0].var(dim=0).mean().cpu()))
            if done or truncated:
                break
        errors.append(seq_errors[-1] if seq_errors else 0.0)
        disagreements.append(float(np.mean(seq_dis)) if seq_dis else 0.0)
    restore_state(env, start_state)
    return float(np.mean(errors)), float(np.mean(disagreements))


def evaluate_prefix(model, mode, seed, prefix_index, args):
    rng = np.random.default_rng(seed * 1009 + prefix_index * 17 + 7)
    env = PusherEnv()
    obs, _ = reset(env, "pusher", seed)
    actions = prefix_actions(rng, mode, args.prefix_steps, env.action_space.shape[0])
    for action in actions:
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            raise RuntimeError("prefix terminated unexpectedly")
    start_obs = np.asarray(obs, dtype=np.float32).copy()
    start_state = env_state(env)

    # Match the mechanism proposal with a low-cost, fixed random action probe.
    probe = rng.uniform(-2.0, 2.0, size=(args.probe_actions, env.action_space.shape[0])).astype(np.float32)
    disagreement, _ = one_step_disagreement(model, start_obs, probe)
    sequences = rng.uniform(-2.0, 2.0,
                            size=(args.rollout_sequences, args.rollout_horizon,
                                  env.action_space.shape[0])).astype(np.float32)
    error, rollout_dis = rollout_error(model, env, start_obs, start_state, sequences)

    model_env = ModelEnv(env, model, termination_fns.no_termination, reward_fns.pusher,
                         generator=torch.Generator().manual_seed(seed + 10000 + prefix_index))
    cfg = OmegaConf.create(dict(_target_="mbrl.planning.CEMOptimizer", num_iterations=args.iterations,
                                elite_ratio=.1, population_size=args.population, alpha=.1,
                                device="cpu", return_mean_elites=False, clipped_normal=False))
    agent = TrajectoryOptimizerAgent(cfg, env.action_space.low.tolist(), env.action_space.high.tolist(),
                                     planning_horizon=args.horizon, replan_freq=1)
    agent.set_trajectory_eval_fn(lambda initial, candidates:
                                 model_env.evaluate_action_sequences(candidates, initial, args.particles))
    total, steps = 0.0, 0
    obs = start_obs.copy()
    planning_start = time.monotonic()
    try:
        for _ in range(args.max_steps):
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += float(reward); steps += 1
            if done or truncated:
                break
    finally:
        env.close()
    return dict(mode=mode, seed=seed, prefix_index=prefix_index,
                prefix_steps=args.prefix_steps, return_=total, steps=steps,
                one_step_ensemble_disagreement=disagreement,
                rollout_ensemble_disagreement=rollout_dis,
                rollout_rmse=error, planning_seconds=time.monotonic()-planning_start)


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    files = ["scripts/pilot_mbrl_provenance_shadow.py", "scripts/pilot_mbrl_rank_general.py"]
    save(out / "plan.json", dict(arguments=vars(args),
         hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
         operation="Legal boundary-state prefixes before native Pusher CEM; measure low ensemble variance versus truthful rollout error and return.",
         controls="clean reset, random legal prefix, interior legal prefix; no model/planner/observation/reward modification.",
         promotion_gate="Reject if boundary prefixes do not show low-variance/high-error decoupling or do not reduce native return relative to clean/interior controls.",
         api_requests=0, gpu_work=False))
    model = make_model("pusher")
    model.load(Path(args.model)); model.eval()
    digest = model_hash(model)
    rows = []
    started = time.monotonic()
    for seed in args.seeds:
        rows.append(evaluate_prefix(model, "clean", seed, 0, args))
        for mode in ("random", "interior", "boundary"):
            for i in range(args.prefixes):
                row = evaluate_prefix(model, mode, seed, i, args)
                rows.append(row)
                print(json.dumps(row), flush=True)
    assert model_hash(model) == digest
    save(out / "results.json", rows)
    grouped = {}
    for mode in ["clean", "random", "interior", "boundary"]:
        values = [r for r in rows if r["mode"] == mode]
        grouped[mode] = {"n": len(values),
                         "return_mean": float(np.mean([r["return_"] for r in values])),
                         "return_std": float(np.std([r["return_"] for r in values])),
                         "one_step_disagreement_mean": float(np.mean([r["one_step_ensemble_disagreement"] for r in values])),
                         "rollout_disagreement_mean": float(np.mean([r["rollout_ensemble_disagreement"] for r in values])),
                         "rollout_rmse_mean": float(np.mean([r["rollout_rmse"] for r in values]))}
    result = dict(status="completed", model_hash=digest, rows=rows, grouped=grouped,
                  elapsed_seconds=time.monotonic()-started, api_requests=0, gpu_work=False,
                  decision="screen_only_pending_gate")
    save(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--prefixes", type=int, default=5); p.add_argument("--prefix-steps", type=int, default=8)
    p.add_argument("--probe-actions", type=int, default=64); p.add_argument("--rollout-sequences", type=int, default=8)
    p.add_argument("--rollout-horizon", type=int, default=15); p.add_argument("--particles", type=int, default=10)
    p.add_argument("--population", type=int, default=50); p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--horizon", type=int, default=15); p.add_argument("--max-steps", type=int, default=25)
    main(p.parse_args())
