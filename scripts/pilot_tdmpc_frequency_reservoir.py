"""Short native gate for a frequency-selective shared-representation hypothesis.

The source contributor executes only legal simulator actions.  The low-frequency
and high-frequency arms have the same nominal amplitude envelope; the latter is
the spectral control.  Each arm uses a fresh official multitask TD-MPC2
checkpoint and the upstream full update path.  This is a candidate screen, not
evidence for a broad claim until the effect survives a second task family.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, episode, set_seed
from pilot_tdmpc_adaptation import adapt, frozen_digest
from screen_tdmpc_non_normal import one_step_error


def _source_action(t, dim, mode, total):
    phase = torch.arange(dim, dtype=torch.float32) * (2.0 * math.pi / max(dim, 1))
    if mode == 'low_freq':
        # Four to eight cycles over a 300-step source episode, with equal
        # per-joint phase offsets so the source remains a legal smooth gait.
        envelope = 0.72 + 0.08 * math.sin(2.0 * math.pi * t / max(total, 1))
        return envelope * torch.sin(2.0 * math.pi * t / 48.0 + phase)
    if mode == 'high_freq':
        # Same amplitude envelope and per-joint phase offsets, but a period-4
        # actuator sequence.  This is the predeclared spectral control.
        envelope = 0.72 + 0.08 * math.sin(2.0 * math.pi * t / max(total, 1))
        return envelope * torch.sin(2.0 * math.pi * t / 4.0 + phase)
    if mode == 'random':
        return torch.rand(dim) * 2.0 - 1.0
    if mode == 'steady':
        # Legal stationary source experience for the semigroup-defect screen.
        return torch.zeros(dim)
    raise ValueError(mode)


def collect(agent, task_name, seed, steps, mode):
    set_seed(seed)
    cfg = agent.cfg
    task = cfg.tasks.index(task_name)
    env = make_one(task_name, seed)
    obs = pad_obs(env.reset(), cfg)
    observations, actions, rewards = [obs.clone()], [], []
    try:
        for t in range(steps):
            raw = _source_action(t, cfg.action_dims[task], mode, steps)
            action = torch.zeros(cfg.action_dim)
            action[:cfg.action_dims[task]] = torch.clamp(raw, -1.0, 1.0)
            nxt, reward, done, _ = env.step(action[:cfg.action_dims[task]])
            observations.append(pad_obs(nxt, cfg)); actions.append(action)
            rewards.append(float(reward)); obs = observations[-1]
            if done:
                break
    finally:
        env.unwrapped.close()
    return {'obs': torch.stack(observations), 'action': torch.stack(actions),
            'reward': torch.tensor(rewards), 'task': task, 'seed': seed}


def spectrum(data):
    x = data['action'][:, :].numpy()
    centered = x - x.mean(axis=0, keepdims=True)
    fft = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0)
    low = float(fft[(freqs > 0) & (freqs <= 1 / 16)].sum())
    high = float(fft[freqs >= 1 / 8].sum())
    return {'low_band_power': low, 'high_band_power': high,
            'low_high_ratio': low / max(high, 1e-12),
            'action_rms': float(np.sqrt(np.mean(x * x))),
            'action_abs_mean': float(np.mean(np.abs(x)))}


def action_jacobian(agent, data, task_name, count=2):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    obs = data['obs'].to(agent.device)
    acts = data['action'].to(agent.device)
    ids = torch.linspace(0, len(acts) - 1, count).long().tolist()
    rows = []
    for i in ids:
        with torch.no_grad():
            z = model.encode(obs[i].unsqueeze(0), task)[0]
        a = acts[i].detach().clone().requires_grad_(True)
        def f(x):
            return model.next(z.unsqueeze(0), x.unsqueeze(0), task)[0]
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f, a, create_graph=False)
        sv = torch.linalg.svdvals(jac)
        rows.append({'index': int(i), 'largest_sv': float(sv.max().detach().cpu()),
                     'smallest_sv': float(sv.min().detach().cpu()),
                     'condition_number': float((sv.max() / (sv.min() + 1e-8)).detach().cpu()),
                     'effective_rank': int((sv > sv.max() * 1e-3).sum().detach().cpu())})
    return {'task': task_name, 'rows': rows,
            'means': {k: float(np.mean([r[k] for r in rows]))
                      for k in ['largest_sv', 'smallest_sv', 'condition_number', 'effective_rank']}}


def three_step_defect(agent, data, task_name):
    """Native three-step composition error against the encoded real endpoint."""
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    obs, acts = data['obs'].to(agent.device), data['action'].to(agent.device)
    if len(acts) < 3:
        return float('nan')
    with torch.no_grad():
        z = model.encode(obs[:-3], task)
        for k in range(3):
            z = model.next(z, acts[k:len(acts)-2+k], task)
        target = model.encode(obs[3:], task)
        return float(torch.mean((z - target).square()).detach().cpu())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a fresh output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    save(out / 'plan.json', {
        **vars(args),
        'operation': 'Legal stationary source actions versus random control; native TD-MPC2 full update; planner unchanged.',
        'hypothesis': 'Valid stationary experience can create a semigroup/three-step composition defect while preserving target one-step prediction, selectively harming native MPC rather than the actor.',
        'promotion_gate': 'Require stationary arm to reduce native MPC return on both seeds, retain one-step error <=1.05x random, increase three-step defect >=1.25x random, and preserve actor return. Otherwise reject.',
        'api_requests': 0,
        'gpu_work': 'bounded native inference and full-update adaptation only'
    })
    results = []
    started = time.monotonic()
    for condition in ['clean', 'random', 'steady']:
        for seed in args.eval_seeds:
            agent, provenance = load_agent(str(checkpoint))
            if condition == 'clean':
                source_data = None
            else:
                source_data = collect(agent, args.source_task, args.source_seed, args.source_steps, condition)
                torch.save(source_data, out / f'{condition}_source.pt')
                before = frozen_digest(agent.model)
                history = adapt(agent, source_data, args.updates, seed=args.learning_seed, scope='native_full')
                save(out / f'{condition}_seed{seed}_adaptation.json', {
                    'updates': history, 'non_dynamics_digest_before': before,
                    'non_dynamics_digest_after': frozen_digest(agent.model),
                    'source_spectrum': spectrum(source_data),
                })
            validation = collect(agent, args.target_task, args.validation_seed, args.eval_steps, 'random')
            mpc, _ = episode(agent, args.target_task, seed, policy='mpc', steps=args.eval_steps)
            actor, _ = episode(agent, args.target_task, seed, policy='prior', steps=args.eval_steps)
            row = {
                'condition': condition, 'target_task': args.target_task, 'eval_seed': seed,
                'mpc_return': mpc['return'], 'actor_return': actor['return'],
                'one_step_error': one_step_error(agent, validation, args.target_task),
                'three_step_defect': three_step_defect(agent, validation, args.target_task),
                'action_jacobian': action_jacobian(agent, validation, args.target_task, args.jacobian_states),
                'source_spectrum': None if source_data is None else spectrum(source_data),
                'updates': 0 if source_data is None else args.updates,
                'wall_seconds': time.monotonic() - started,
            }
            results.append(row); save(out / 'results.json', results); print(json.dumps(row), flush=True)
    by = {(r['condition'], r['eval_seed']): r for r in results}
    gates = []
    for seed in args.eval_seeds:
        steady, rnd = by[('steady', seed)], by[('random', seed)]
        gates.append({
            'seed': seed,
            'steady_vs_random_loss': 1.0 - steady['mpc_return'] / max(rnd['mpc_return'], 1e-8),
            'steady_error_ratio_vs_random': steady['one_step_error'] / max(rnd['one_step_error'], 1e-12),
            'three_step_defect_ratio_vs_random': steady['three_step_defect'] / max(rnd['three_step_defect'], 1e-12),
            'actor_delta_steady_vs_random': steady['actor_return'] - rnd['actor_return'],
            'jacobian_condition_ratio_steady_vs_random': steady['action_jacobian']['means']['condition_number'] / max(rnd['action_jacobian']['means']['condition_number'], 1e-12),
            'passed': (steady['mpc_return'] <= 0.85 * rnd['mpc_return'] and
                       steady['one_step_error'] <= 1.05 * rnd['one_step_error'] and
                       steady['three_step_defect'] >= 1.25 * rnd['three_step_defect'] and
                       abs(steady['actor_return'] - rnd['actor_return']) <= 0.05 * max(abs(rnd['actor_return']), 1.0))
        })
    status = {
        'status': 'completed', 'results': results, 'gates': gates,
        'all_passed': all(g['passed'] for g in gates),
        'decision': 'Only a short mechanism screen. Do not promote or scale unless all seed gates pass and a second target family confirms the same spectral selectivity.',
        'elapsed_seconds': time.monotonic() - started, 'api_requests': 0,
    }
    save(out / 'status.json', status); print(json.dumps(status), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-task', default='cheetah-run')
    p.add_argument('--target-task', default='walker-run')
    p.add_argument('--source-seed', type=int, default=23)
    p.add_argument('--validation-seed', type=int, default=33)
    p.add_argument('--eval-seeds', nargs='+', type=int, default=[41, 42])
    p.add_argument('--source-steps', type=int, default=300)
    p.add_argument('--eval-steps', type=int, default=300)
    p.add_argument('--updates', type=int, default=80)
    p.add_argument('--learning-seed', type=int, default=31)
    p.add_argument('--jacobian-states', type=int, default=2)
    p.add_argument('--output', required=True)
    main(p.parse_args())
