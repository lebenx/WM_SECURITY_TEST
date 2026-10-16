"""Search for an execution-indistinguishable counterfactual-query side channel.

The service sees honest latent/action requests emitted by a native TD-MPC2 CEM
client, but not the client's private terminal constraint.  The search does not
alter the model or environment.  It only asks whether rejected counterfactual
queries carry more information about that constraint than the first executed
action does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed


def plan(agent, z0, goal, task, seed, private_weight, horizon=5, samples=96,
         iterations=4):
    set_seed(seed)
    cfg = agent.cfg
    mask = agent.model._action_masks[task]
    mean = torch.zeros(horizon, cfg.action_dim, device=agent.device)
    std = torch.full_like(mean, 0.8)
    query = []
    public = torch.zeros(samples, device=agent.device)
    for _ in range(iterations):
        actions = (mean[:, None] + std[:, None] * torch.randn(
            horizon, samples, cfg.action_dim, device=agent.device)).clamp(-1, 1)
        actions *= mask
        z = z0.expand(samples, -1)
        for t in range(horizon):
            query.append(torch.cat([z.detach().cpu(), actions[t].detach().cpu()], 1))
            public = public + (0.99 ** t) * agent.model.reward(
                z, actions[t], task).mean(-1)
            z = agent.model.next(z, actions[t], task)
        private = (z.detach().cpu() - goal.cpu()).square().mean(1).to(agent.device)
        score = public - private_weight * private
        elite = torch.topk(score, min(16, samples)).indices
        mean = actions[:, elite].mean(1)
        std = actions[:, elite].std(1).clamp(0.08, 0.8)
    query = torch.cat(query).numpy()
    # The service may compute moments but does not receive the private goal.
    # Include low-order latent and action geometry, plus quantiles, without
    # using the private objective or returned rewards.
    summary = []
    for part in (query[:, :cfg.latent_dim], query[:, cfg.latent_dim:]):
        summary.extend(part.mean(0)); summary.extend(part.std(0))
        summary.extend(np.quantile(part, [.1, .5, .9], axis=0).reshape(-1))
    return {
        'first_action': mean[0].detach().cpu().numpy(),
        'summary': np.asarray(summary, dtype=np.float32),
        'query_count': int(query.shape[0]),
    }


def candidate_search(agent, z0, task, rng, weight, goals=24, pairs=180):
    latent = agent.cfg.latent_dim
    gs = [torch.tensor(rng.normal(0, .35, latent), dtype=torch.float32)
          for _ in range(goals)]
    best = None
    for i in range(pairs):
        a, b = rng.integers(0, goals, 2)
        if a == b:
            continue
        ra = plan(agent, z0, gs[a], task, 900000 + i, weight)
        rb = plan(agent, z0, gs[b], task, 900000 + i, weight)
        action_diff = float(np.abs(ra['first_action'] - rb['first_action']).mean())
        # Standardised query distance is only a ranking signal for selecting a
        # pair; final validation uses held-out planner seeds and a classifier.
        scale = np.maximum(np.abs(ra['summary']) + np.abs(rb['summary']), 1e-3)
        query_distance = float(np.mean(np.abs(ra['summary'] - rb['summary']) / scale))
        score = query_distance / (action_diff + .01)
        row = dict(a=int(a), b=int(b), action_diff=action_diff,
                   query_distance=query_distance, score=score)
        if best is None or row['score'] > best['score']:
            best = dict(row, goal_a=gs[a], goal_b=gs[b])
    return best


def validate(agent, z0, task, best, weight, reps=16):
    rows = []
    for label, goal in enumerate([best['goal_a'], best['goal_b']]):
        for rep in range(reps):
            r = plan(agent, z0, goal, task, 910000 + rep, weight)
            rows.append({'label': label, 'rep': rep,
                         'first_action': r['first_action'].tolist(),
                         'summary': r['summary'].tolist()})
    x = np.asarray([r['summary'] for r in rows])
    y = np.asarray([r['label'] for r in rows])
    pred = []
    for i in range(len(rows)):
        train = np.arange(len(rows)) != i
        centers = {c: x[train & (y == c)].mean(0) for c in np.unique(y)}
        pred.append(min(centers, key=lambda c: float(np.mean((x[i] - centers[c]) ** 2))))
    action = np.asarray([r['first_action'] for r in rows])
    paired = float(np.mean(np.abs(action[:reps] - action[reps:])))
    return dict(rows=rows, accuracy=float(np.mean(np.asarray(pred) == y)),
                paired_first_action_mean_abs_difference=paired)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--tasks', nargs='+', default=['walker-walk', 'cup-catch', 'reacher-easy'])
    p.add_argument('--weight', type=float, default=.20)
    p.add_argument('--goals', type=int, default=24)
    p.add_argument('--pairs', type=int, default=180)
    p.add_argument('--reps', type=int, default=16)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=False, exist_ok=False)
    rng = np.random.default_rng(20261010)
    agent, provenance = load_agent(Path('models/tdmpc2/mt30-48M.pt'))
    agent.model.requires_grad_(False)
    all_results = []
    started = time.monotonic()
    for task_name in args.tasks:
        task = agent.cfg.tasks.index(task_name)
        env = make_one(task_name, 31)
        obs = pad_obs(env.reset(), agent.cfg)
        env.unwrapped.close()
        with torch.no_grad():
            z0 = agent.model.encode(obs.to(agent.device).unsqueeze(0),
                                    torch.tensor([task], device=agent.device))
        best = candidate_search(agent, z0, task, rng, args.weight, args.goals, args.pairs)
        check = validate(agent, z0, task, best, args.weight, args.reps)
        result = {'task': task_name, 'weight': args.weight,
                  'search': {k: v for k, v in best.items() if k not in ('goal_a', 'goal_b')},
                  'validation': {k: v for k, v in check.items() if k != 'rows'},
                  'goals': [best['goal_a'].tolist(), best['goal_b'].tolist()],
                  'validation_rows': check['rows']}
        all_results.append(result)
        print(json.dumps({k: result[k] for k in ('task', 'search', 'validation')}, ensure_ascii=False), flush=True)
    plan = {'status': 'completed', 'tasks': args.tasks, 'weight': args.weight,
            'threat_model': 'honest TD-MPC2 transition service observes latent/action query geometry but not private terminal goal; no response tampering or model updates',
            'search': {'goals': args.goals, 'pairs_per_task': args.pairs, 'validation_repetitions': args.reps},
            'checkpoint': 'models/tdmpc2/mt30-48M.pt',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'results': all_results, 'elapsed_seconds': time.monotonic() - started,
            'api_requests': 0, 'gpu_training': False}
    (out / 'summary.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'status': plan['status'], 'elapsed_seconds': plan['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
