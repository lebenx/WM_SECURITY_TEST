"""Experimental horizon switching using fixed-state Jacobian powers.

This local proxy is not the Jacobian product of an imagined trajectory and
is not specific to non-normality. Earlier runs used a non-native planner and
reset its warm start every step; those runs cannot evaluate this defense.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


@torch.no_grad()
def _latent(agent, obs, task):
    return agent.model.encode(obs.to(agent.device).unsqueeze(0), torch.tensor([task], device=agent.device))[0]


def power_probe(agent, obs, task, horizon=5):
    """max_k ||J(z, zero_action)^k||_2; fixed local Jacobian, no RNG use."""
    cfg, model = agent.cfg, agent.model
    task_t = torch.tensor([task], device=agent.device)
    z0 = _latent(agent, obs, task).detach()
    action = torch.zeros(cfg.action_dim, device=agent.device)
    def f(x):
        return model.next(x.unsqueeze(0), action.unsqueeze(0), task_t)[0]
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(f, z0, create_graph=False)
    power = J
    best = 0.0
    for _ in range(horizon):
        best = max(best, float(torch.linalg.matrix_norm(power, ord=2).detach().cpu()))
        power = power @ J
    return best


def set_horizon(agent, horizon):
    if agent.cfg.horizon == horizon:
        return
    old = agent._prev_mean
    new = torch.zeros(horizon, agent.cfg.action_dim, device=agent.device)
    count = min(horizon, old.shape[0])
    new[:count].copy_(old[:count])
    agent.cfg.horizon = horizon
    agent._prev_mean = torch.nn.Buffer(new)


def selective_episode(agent, task_name, seed, steps, threshold, probe_every):
    cfg, task = agent.cfg, agent.cfg.tasks.index(task_name)
    set_seed(seed)
    env = make_one(task_name, seed)
    obs = pad_obs(env.reset(), cfg)
    rewards, probes, horizons = [], [], []
    old_horizon = cfg.horizon
    old_prev = agent._prev_mean.clone()
    try:
        for step in range(steps):
            if step % probe_every == 0:
                metric = power_probe(agent, obs, task, horizon=5)
                probes.append({'step': step, 'power_probe': metric})
            else:
                metric = probes[-1]['power_probe']
            horizon = 1 if metric > threshold else old_horizon
            set_horizon(agent, horizon)
            action = agent.act(obs, t0=step == 0, task=task, eval_mode=False)
            next_obs, reward, done, _ = env.step(action[:cfg.action_dims[task]])
            obs = pad_obs(next_obs, cfg)
            rewards.append(float(reward)); horizons.append(horizon)
            if done:
                break
    finally:
        cfg.horizon = old_horizon
        agent._prev_mean = torch.nn.Buffer(old_prev)
        env.unwrapped.close()
    return {
        'task': task_name, 'seed': seed, 'steps': len(rewards),
        'return': sum(rewards), 'full_episode': len(rewards) == 500, 'reached_cap': len(rewards) == steps,
        'threshold': threshold, 'probe_every': probe_every,
        'horizon1_steps': int(sum(h == 1 for h in horizons)),
        'horizon1_fraction': float(sum(h == 1 for h in horizons) / max(1, len(horizons))),
        'probe_rows': probes,
    }


def main(args):
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'plan.json', {
        **vars(args),
        'operation': 'Fixed-state Jacobian powers at zero action; native MPPI switches from its configured horizon to 1. No-op switching preserves warm start.',
        'controls': 'Same adapted cascade and random arms as dose-300 gate; no adaptive threshold tuning from victim returns.',
        'api_requests': 0,
    })
    rows = []
    for label, source_path, victim_task, eval_seed in args.arm:
        agent, provenance = load_agent(args.checkpoint)
        source = torch.load(ROOT / source_path, weights_only=False)
        history = adapt(agent, source, args.updates, seed=31, scope='dynamics')
        result = selective_episode(agent, victim_task, eval_seed, args.eval_steps,
                                   args.threshold, args.probe_every)
        row = {'arm': label, **result, 'source_steps': len(source['action']),
               'updates': args.updates,
               'final_total_loss': history[-1].get('weighted_consistency_loss', history[-1].get('total_loss'))}
        rows.append(row); print(json.dumps(row), flush=True)
        save(out / f'{label}_adaptation.json', {'source': source_path, 'history': history, 'provenance': provenance})
        save(out / 'results.json', rows)
    save(out / 'status.json', {
        'status': 'completed', 'results': rows,
        'decision': 'Promote only if selective horizon recovers cascade returns while preserving random-control and clean returns with a low trigger rate.',
    })


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--eval-steps', type=int, default=300)
    p.add_argument('--threshold', type=float, default=16.0)
    p.add_argument('--probe-every', type=int, default=10)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', nargs=4, action='append', metavar=('LABEL', 'SOURCE', 'TASK', 'SEED'), required=True)
    a = p.parse_args()
    a.arm = [(x[0], x[1], x[2], int(x[3])) for x in a.arm]
    main(a)
