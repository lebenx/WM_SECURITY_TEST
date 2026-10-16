"""Short gate for the proposed legal-cascade/non-normal dynamics mechanism.

The source contributor only executes legal cheetah actions.  We adapt the
shared TD-MPC2 dynamics head on either ordinary random data or a monotonic
high-acceleration action curriculum, then measure latent Jacobian metrics and a
held-out native walker controller.  This is a falsification screen, not a
claim of an online deployment protocol.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, episode, set_seed
from pilot_tdmpc_adaptation import adapt


def collect(agent, task, seed, steps, mode, cascade_gain=1.25):
    set_seed(seed)
    idx = agent.cfg.tasks.index(task)
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), agent.cfg)
    observations, actions, rewards = [obs.clone()], [], []
    try:
        for t in range(steps):
            action = torch.zeros(agent.cfg.action_dim)
            dim = agent.cfg.action_dims[idx]
            if mode == 'random':
                action[:dim] = torch.rand(dim) * 2 - 1
            elif mode == 'cascade':
                # A legal monotonic actuator curriculum.  Each joint ramps
                # upward at a different phase, producing a directed cascade.
                phase = torch.arange(dim, dtype=torch.float32) / max(dim, 1)
                level = min(1.0, max(0.0, (t / max(steps - 1, 1)) * cascade_gain))
                action[:dim] = torch.clamp(level + 0.35 * phase, -1., 1.)
            else:
                raise ValueError(mode)
            next_obs, reward, done, _ = env.step(action[:dim])
            observations.append(pad_obs(next_obs, agent.cfg))
            actions.append(action)
            rewards.append(float(reward))
            obs = observations[-1]
            if done:
                break
    finally:
        env.unwrapped.close()
    return {'obs': torch.stack(observations), 'action': torch.stack(actions),
            'reward': torch.tensor(rewards), 'task': idx, 'seed': seed}


def latent_metrics(agent, data, task_name, count=3, horizon=5):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    observations = data['obs'].to(agent.device)
    actions = data['action'].to(agent.device)
    indices = torch.linspace(0, len(actions) - 1, count).long().tolist()
    rows = []
    for i in indices:
        obs = observations[i]
        action = actions[i]
        with torch.no_grad():
            z = model.encode(obs.unsqueeze(0), task)[0]
        def f(x):
            return model.next(x.unsqueeze(0), action.unsqueeze(0), task)[0]
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(f, z, create_graph=False)
        eig = torch.linalg.eigvals(J)
        fro = torch.linalg.matrix_norm(J, ord='fro')
        departure = torch.sqrt(torch.clamp(fro.square() - eig.abs().square().sum(), min=0.)) / (fro + 1e-8)
        powers = []
        for k in range(1, horizon + 1):
            powers.append(float(torch.linalg.matrix_norm(torch.linalg.matrix_power(J, k), ord=2).detach().cpu()))
        rows.append({'index': int(i), 'spectral_radius': float(eig.abs().max().detach().cpu()),
                     'henrici_departure': float(departure.detach().cpu()),
                     'max_power_norm': max(powers), 'power_norms': powers})
    return {'task': task_name, 'rows': rows,
            'means': {key: float(np.mean([r[key] for r in rows]))
                      for key in ['spectral_radius', 'henrici_departure', 'max_power_norm']}}


def one_step_error(agent, data, task_name):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    obs, actions = data['obs'].to(agent.device), data['action'].to(agent.device)
    with torch.no_grad():
        z = model.encode(obs[:-1], task)
        target = model.encode(obs[1:], task)
        pred = model.next(z, actions, task)
        return float(torch.mean((pred - target).square()).cpu())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', vars(args) | {
        'operation': 'Legal monotonic cascade versus matched random source data; dynamics-only adaptation; no edits to observations, rewards, weights or planner.',
        'gate': 'Compare Jacobian spectral radius, Henrici departure, finite-horizon power norm, one-step error, and native walker return.',
        'gpu_work': 'short CUDA inference/training only'
    })
    clean_agent, provenance = load_agent(args.checkpoint)
    save(out / 'provenance.json', provenance)
    walker_data = collect(clean_agent, args.victim_task, args.eval_seed, args.eval_steps, 'random')
    clean_return, _ = episode(clean_agent, args.victim_task, args.eval_seed, steps=args.eval_steps)
    clean_jac = latent_metrics(clean_agent, walker_data, args.victim_task, args.jacobian_states, args.horizon)
    clean_err = one_step_error(clean_agent, walker_data, args.victim_task)
    results = {'clean': {'return': clean_return['return'], 'one_step_error': clean_err, 'jacobian': clean_jac}}
    save(out / 'results.json', results)
    print(json.dumps({'condition': 'clean', **results['clean']}), flush=True)
    for mode in ['random', 'cascade']:
        agent, _ = load_agent(args.checkpoint)
        source = collect(agent, args.source_task, args.source_seed, args.source_steps, mode,
                         cascade_gain=args.cascade_gain)
        torch.save(source, out / f'{mode}_source.pt')
        history = adapt(agent, source, updates=args.updates, seed=31, scope='dynamics')
        save(out / f'{mode}_adaptation.json', {'updates': history})
        heldout = collect(agent, args.victim_task, args.eval_seed, args.eval_steps, 'random')
        ret, _ = episode(agent, args.victim_task, args.eval_seed, steps=args.eval_steps)
        row = {'return': ret['return'], 'one_step_error': one_step_error(agent, heldout, args.victim_task),
               'jacobian_source': latent_metrics(agent, source, args.source_task, args.jacobian_states, args.horizon),
               'jacobian_victim': latent_metrics(agent, heldout, args.victim_task, args.jacobian_states, args.horizon),
               'source_mode': mode, 'source_steps': len(source['action']), 'updates': args.updates}
        results[mode] = row
        save(out / 'results.json', results)
        print(json.dumps({'condition': mode, **row}), flush=True)
    save(out / 'status.json', {'status': 'completed', 'results': results,
                               'decision': 'Screen only. Candidate requires cascade-specific power growth with preserved one-step error and native task harm; no broad claim is inferred here.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-task', choices=['cheetah-run', 'walker-walk-backwards', 'walker-walk', 'hopper-hop'], default='cheetah-run')
    p.add_argument('--cascade-gain', type=float, default=1.25)
    p.add_argument('--source-seed', type=int, default=21)
    p.add_argument('--eval-seed', type=int, default=11)
    p.add_argument('--source-steps', type=int, default=300)
    p.add_argument('--eval-steps', type=int, default=300)
    p.add_argument('--victim-task', choices=['cheetah-run', 'walker-walk', 'walker-run', 'hopper-hop', 'reacher-easy', 'cup-catch'], default='walker-walk')
    p.add_argument('--updates', type=int, default=120)
    p.add_argument('--jacobian-states', type=int, default=2)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--output', required=True)
    main(p.parse_args())
