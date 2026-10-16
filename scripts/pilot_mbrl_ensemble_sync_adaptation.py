"""Short native gate for truthful source-induced ensemble residual synchronization.

This tests a distinct mechanism from inference-time sampler coupling: a legal
source trajectory is used for a small online model update.  Promotion requires
source-specific growth in cross-member residual correlation while target
one-step mean/variance stay matched and native CEM return falls.  Random and
shuffled boundary source controls are included to separate order from dose.
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
from mbrl.types import TransitionBatch
from pilot_mbrl_rank_general import make_model, model_hash, reset


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float) + "\n")


def collect(seed, mode, n, shuffle_seed=0):
    rng = np.random.default_rng(seed * 101 + shuffle_seed)
    env = PusherEnv(); obs, _ = reset(env, "pusher", seed)
    rows = []
    for i in range(n):
        if mode == "random":
            action = rng.uniform(-2.0, 2.0, size=7).astype(np.float32)
        elif mode == "boundary":
            action = rng.choice(np.array([-2.0, 2.0], dtype=np.float32), size=7)
        elif mode == "shuffled_boundary":
            action = rng.choice(np.array([-2.0, 2.0], dtype=np.float32), size=7)
        else:
            raise ValueError(mode)
        nxt, reward, done, truncated, _ = env.step(action)
        rows.append((obs.copy(), action.copy(), np.asarray(nxt, dtype=np.float32).copy(),
                     float(reward), bool(done), bool(truncated)))
        obs = nxt
        if done or truncated:
            obs, _ = reset(env, "pusher", seed + i + 1)
    env.close()
    if mode == "shuffled_boundary":
        rng2 = np.random.default_rng(shuffle_seed + 991)
        perm = rng2.permutation(len(rows))
        rows = [rows[i] for i in perm]
    data = [np.asarray(x, dtype=bool if j >= 4 else np.float32)
            for j, x in enumerate(zip(*rows))]
    return TransitionBatch(*data)


def adapt(model, source, updates, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    losses = []
    for _ in range(updates):
        idx = rng.integers(0, len(source), size=min(64, len(source)))
        batch = source[idx]
        loss, _ = model.update(batch, optimizer)
        losses.append(float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss))
    return losses


@torch.no_grad()
def diagnostics(model, validation):
    obs, actions, nxt, *_ = validation.astuple()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    nxt = torch.as_tensor(nxt, dtype=torch.float32)
    x = model._get_model_input(obs, actions)
    means, _ = model.model.forward(x, use_propagation=False)
    pred = obs.unsqueeze(0) + means
    truth = nxt.unsqueeze(0).expand_as(pred)
    err = pred - truth
    member_rmse = torch.sqrt(err.square().mean(dim=(1, 2)))
    flat = err.reshape(err.shape[0], -1).cpu().numpy()
    corr = np.corrcoef(flat)
    off = corr[np.triu_indices_from(corr, k=1)]
    return dict(one_step_rmse=float(torch.sqrt((pred.mean(0)-nxt).square().mean())),
                predictive_mean_rmse=float(torch.sqrt((pred.mean(0)-nxt).square().mean())),
                ensemble_variance=float(pred.var(dim=0).mean()),
                member_rmse_mean=float(member_rmse.mean()),
                member_rmse_std=float(member_rmse.std()),
                residual_corr_mean=float(np.nanmean(off)),
                residual_corr_min=float(np.nanmin(off)),
                residual_corr_max=float(np.nanmax(off)))


def plan_return(model, seed, args):
    env = PusherEnv(); obs, _ = reset(env, "pusher", seed)
    model_env = ModelEnv(env, model, termination_fns.no_termination, reward_fns.pusher,
                         generator=torch.Generator().manual_seed(seed + 10000))
    cfg = OmegaConf.create(dict(_target_="mbrl.planning.CEMOptimizer", num_iterations=2,
                                elite_ratio=.1, population_size=50, alpha=.1, device="cpu",
                                return_mean_elites=False, clipped_normal=False))
    agent = TrajectoryOptimizerAgent(cfg, env.action_space.low.tolist(), env.action_space.high.tolist(),
                                     planning_horizon=args.horizon, replan_freq=1)
    agent.set_trajectory_eval_fn(lambda initial, candidates:
                                 model_env.evaluate_action_sequences(candidates, initial, args.particles))
    total = 0.0; steps = 0
    try:
        for _ in range(args.max_steps):
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += float(reward); steps += 1
            if done or truncated: break
    finally:
        env.close()
    return dict(return_=total, steps=steps)


def main(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/pilot_mbrl_ensemble_sync_adaptation.py", "scripts/pilot_mbrl_rank_general.py"]
    save(out / "plan.json", dict(arguments=vars(args), hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
         operation="Legal Pusher source transitions update all five GaussianMLP members; measure target residual correlation and native CEM.",
         promotion_gate="Require boundary-vs-random/shuffled residual-correlation increase with <=5% one-step mean/variance change and >=40% CEM return loss on two seeds.",
         api_requests=0, gpu_work=False))
    base = make_model("pusher"); base.load(Path(args.model)); base.eval()
    base_hash = model_hash(base)
    validation = collect(900, "random", args.validation_steps, 909)
    rows = []; started = time.monotonic()
    for condition in ("clean", "random", "boundary", "shuffled_boundary"):
        model = make_model("pusher"); model.load(Path(args.model)); model.eval()
        losses = []
        if condition != "clean":
            source = collect(args.source_seed, condition, args.source_steps, args.shuffle_seed)
            losses = adapt(model, source, args.updates, args.learning_seed)
        diag = diagnostics(model, validation)
        returns = [plan_return(model, seed, args) for seed in args.eval_seeds]
        row = dict(condition=condition, updates=args.updates, source_steps=0 if condition == "clean" else args.source_steps,
                   adaptation_loss_tail=losses[-5:], diagnostics=diag, eval_seeds=args.eval_seeds,
                   returns=returns, mean_return=float(np.mean([x['return_'] for x in returns])),
                   model_hash=model_hash(model))
        rows.append(row); save(out / (condition + ".json"), row); print(json.dumps(row), flush=True)
    result = dict(status="completed", base_model_hash=base_hash, rows=rows,
                  elapsed_seconds=time.monotonic()-started, api_requests=0, gpu_work=False,
                  decision="screen_only_pending_gate")
    save(out / "summary.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--output", required=True)
    p.add_argument("--source-seed", type=int, default=31); p.add_argument("--validation-steps", type=int, default=500)
    p.add_argument("--source-steps", type=int, default=120); p.add_argument("--updates", type=int, default=20)
    p.add_argument("--learning-seed", type=int, default=31); p.add_argument("--shuffle-seed", type=int, default=77)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--particles", type=int, default=10); p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=25)
    main(p.parse_args())
