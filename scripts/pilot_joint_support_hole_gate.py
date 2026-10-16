"""Short falsification gate for conditional action-support holes.

The data are truthful transitions from a legal contextual system.  The attack
does not alter observations, actions, rewards, or labels: it chooses actions
whose sign agrees with an observed context.  Context and action marginals stay
balanced, while half of the context-action cells are absent.  This is a
mechanism screen, not a benchmark claim; a native MBRL carrier is required
before any scale-up.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def truth(x, c, a):
    # A smooth, action-dependent interaction.  The attack observes only the
    # c*a >= 0 quadrants, so the missing quadrants remain legal but untrained.
    return x + a + 0.85 * c * np.square(a)


def make_data(n, seed, mode):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.25, 0.25, n).astype(np.float32)
    c = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    if mode == 'clean':
        a = rng.uniform(-1., 1., n).astype(np.float32)
    elif mode == 'support_hole':
        # Both c and a remain balanced.  Only their joint support changes.
        a = (c * rng.uniform(0., 1., n)).astype(np.float32)
    elif mode == 'permuted_control':
        a = rng.uniform(0., 1., n).astype(np.float32)
        a *= rng.choice(np.array([-1., 1.], dtype=np.float32), n)
        rng.shuffle(a)
    else:
        raise ValueError(mode)
    y = truth(x, c, a).astype(np.float32)
    return torch.from_numpy(np.column_stack([x, c, a])), torch.from_numpy(y[:, None])


def train(x, y, seed, steps=900):
    torch.manual_seed(seed)
    model = Dynamics()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    for _ in range(steps):
        pred = model(x)
        loss = (pred - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def evaluate(model, seed):
    rng = np.random.default_rng(seed + 10000)
    # Evaluate separately on observed and missing quadrants.  x is independent
    # of c and action, making the state marginal identical by construction.
    n = 4000
    x = rng.uniform(-0.25, 0.25, n).astype(np.float32)
    c = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    mag = rng.uniform(0.0, 1.0, n).astype(np.float32)
    signs = np.where(rng.random(n) < 0.5, 1., -1.).astype(np.float32)
    a = c * mag * signs
    observed = c * a >= 0
    inp = torch.from_numpy(np.column_stack([x, c, a]))
    with torch.no_grad():
        pred = model(inp).numpy()[:, 0]
    actual = truth(x, c, a)
    mse_observed = float(np.mean((pred[observed] - actual[observed]) ** 2))
    mse_missing = float(np.mean((pred[~observed] - actual[~observed]) ** 2))

    # Native-style one-step action selection over legal actions.  The true goal
    # is a positive displacement; c=-1 requires a positive (missing) action.
    action_grid = np.linspace(-1., 1., 401, dtype=np.float32)
    rows = []
    for context in [-1., 1.]:
        state = np.zeros_like(action_grid)
        contexts = np.full_like(action_grid, context)
        query = torch.from_numpy(np.column_stack([state, contexts, action_grid]))
        with torch.no_grad():
            imagined = model(query).numpy()[:, 0]
        # Both contexts can reach this target with a positive action, but the
        # c=-1 positive quadrant is absent under the attack.
        target = 0.25
        chosen = float(action_grid[np.argmin(np.abs(imagined - target))])
        true_next = float(truth(np.array([0.], dtype=np.float32),
                                np.array([context], dtype=np.float32),
                                np.array([chosen], dtype=np.float32))[0])
        rows.append({'context': context, 'chosen_action': chosen,
                     'true_next': true_next, 'goal_abs_error': abs(true_next-target)})
    return {
        'mse_observed': mse_observed,
        'mse_missing': mse_missing,
        'missing_to_observed_mse_ratio': mse_missing / max(mse_observed, 1e-12),
        'planner_rows': rows,
        'planner_goal_abs_error_mean': float(np.mean([r['goal_abs_error'] for r in rows])),
        'planner_goal_abs_error_c_minus': rows[0]['goal_abs_error'],
    }


def main(args):
    started = time.monotonic()
    all_rows = []
    for seed in args.seeds:
        for mode in ['clean', 'support_hole', 'permuted_control']:
            x, y = make_data(args.samples, seed, mode)
            model = train(x, y, seed + {'clean': 0, 'support_hole': 100, 'permuted_control': 200}[mode],
                          steps=args.steps)
            row = {'seed': seed, 'mode': mode, 'n': args.samples,
                   'context_mean': float(x[:, 1].mean()),
                   'action_mean': float(x[:, 2].mean()),
                   'action_abs_mean': float(x[:, 2].abs().mean()),
                   'observed_joint_fraction': float((x[:, 1] * x[:, 2] >= 0).float().mean()),
                   **evaluate(model, seed)}
            all_rows.append(row)
            print(json.dumps(row), flush=True)
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row['mode'], []).append(row)
    summary = {
        'status': 'completed_short_gate',
        'candidate': 'conditional_action_support_hole',
        'rows': all_rows,
        'means': {mode: {
            'mse_observed': float(np.mean([r['mse_observed'] for r in rows])),
            'mse_missing': float(np.mean([r['mse_missing'] for r in rows])),
            'missing_to_observed_mse_ratio': float(np.mean([r['missing_to_observed_mse_ratio'] for r in rows])),
            'planner_goal_abs_error_mean': float(np.mean([r['planner_goal_abs_error_mean'] for r in rows])),
            'planner_goal_abs_error_c_minus': float(np.mean([r['planner_goal_abs_error_c_minus'] for r in rows])),
            'context_mean': float(np.mean([r['context_mean'] for r in rows])),
            'action_mean': float(np.mean([r['action_mean'] for r in rows])),
            'action_abs_mean': float(np.mean([r['action_abs_mean'] for r in rows])),
        } for mode, rows in grouped.items()},
        'checks': {
            'marginals_balanced': all(abs(r['context_mean']) < .06 and abs(r['action_mean']) < .06 for r in all_rows),
            'support_hole_observed_fraction_near_one': all(
                r['observed_joint_fraction'] > .98 for r in all_rows if r['mode'] == 'support_hole'),
        },
        'scope': 'Synthetic contextual dynamics falsification gate; no native environment, no attack claim, no scale-up.',
        'seeds': args.seeds,
        'samples': args.samples,
        'steps': args.steps,
        'elapsed_s': time.monotonic() - started,
    }
    # Replace the placeholder with a transparent, post-hoc threshold check.
    clean_obs = summary['means']['clean']['mse_observed']
    attack_obs = summary['means']['support_hole']['mse_observed']
    clean_plan = summary['means']['clean']['planner_goal_abs_error_mean']
    attack_plan = summary['means']['support_hole']['planner_goal_abs_error_mean']
    summary['checks']['factual_fit_ratio_attack_to_clean'] = attack_obs / max(clean_obs, 1e-12)
    summary['checks']['planner_error_ratio_attack_to_clean'] = attack_plan / max(clean_plan, 1e-12)
    summary['checks']['candidate_gate_pass'] = bool(
        summary['checks']['marginals_balanced']
        and summary['checks']['support_hole_observed_fraction_near_one']
        and summary['means']['support_hole']['missing_to_observed_mse_ratio'] > 2.0
        and summary['checks']['factual_fit_ratio_attack_to_clean'] < 1.15
        and summary['checks']['planner_error_ratio_attack_to_clean'] > 1.2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'candidate_gate_pass': summary['checks']['candidate_gate_pass'],
                      'output': str(out), 'elapsed_s': summary['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/joint_support_hole_gate_v0/summary.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 12, 13])
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--steps', type=int, default=900)
    main(parser.parse_args())
