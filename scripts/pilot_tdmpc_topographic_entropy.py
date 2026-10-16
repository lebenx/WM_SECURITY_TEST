"""Short screen for categorical-support entropy along native latent rollouts.

The candidate asks whether legal near-zero-reward source experience can flatten
TD-MPC2's categorical reward support along imagined horizons while preserving
one-step dynamics.  The native planner and upstream full update stay unchanged;
this is a falsification screen, not a promoted attack.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt, frozen_digest
from pilot_tdmpc_frequency_reservoir import collect
from screen_tdmpc_non_normal import one_step_error


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


@torch.no_grad()
def reward_entropy(agent, validation, task_name, samples=128):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    obs = validation['obs'].to(agent.device)
    # Fixed validation states and common random candidate actions make the
    # entropy ratio comparable across source conditions.
    states = obs[::max(1, len(obs) // 8)][:8]
    g = torch.Generator(device=agent.device).manual_seed(20261010)
    action = torch.rand(3, len(states), samples, cfg.action_dim,
                        device=agent.device, generator=g) * 2 - 1
    action *= model._action_masks[task]
    z = model.encode(states, task).unsqueeze(1).expand(-1, samples, -1)
    rows = []
    for depth in range(3):
        zflat = z.reshape(-1, cfg.latent_dim)
        aflat = action[depth].transpose(0, 1).reshape(-1, cfg.action_dim)
        logits = model.reward(zflat, aflat, task)
        p = logits.softmax(-1)
        h = -(p * (p.clamp_min(1e-12).log())).sum(-1)
        rows.append(float(h.mean().cpu()))
        z = model.next(zflat, aflat, task).reshape(len(states), samples, -1)
    return {'depth_entropy': rows, 'mean_entropy': float(np.mean(rows))}


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan = dict(vars(args),
                source_mode='steady versus random legal source actions',
                hypothesis='Near-zero-reward legal source experience flattens categorical reward support along imagined horizon while preserving one-step dynamics and selectively harming native MPC.',
                gate='steady must reduce MPC return on both seeds, entropy ratio >=1.10, one-step error <=1.05x random, and actor delta within 5%; otherwise reject.',
                native_planner_unchanged=True, api_requests=0)
    save(out / 'plan.json', plan)
    results = []
    for condition in ['clean', 'random', 'steady']:
        for seed in args.eval_seeds:
            agent, provenance = load_agent(args.checkpoint)
            if condition == 'clean':
                source = None
            else:
                source = collect(agent, args.source_task, args.source_seed,
                                 args.source_steps, condition)
                torch.save(source, out / f'{condition}_source.pt')
                before = frozen_digest(agent.model)
                history = adapt(agent, source, args.updates, args.learning_seed,
                                scope='native_full')
                save(out / f'{condition}_seed{seed}_adaptation.json',
                     {'updates': history, 'non_dynamics_digest_before': before,
                      'non_dynamics_digest_after': frozen_digest(agent.model)})
            validation = collect(agent, args.target_task, args.validation_seed,
                                 args.eval_steps, 'random')
            mpc, _ = episode(agent, args.target_task, seed, policy='mpc',
                              steps=args.eval_steps)
            actor, _ = episode(agent, args.target_task, seed, policy='prior',
                               steps=args.eval_steps)
            row = dict(condition=condition, seed=seed,
                       mpc_return=mpc['return'], actor_return=actor['return'],
                       one_step_error=one_step_error(agent, validation,
                                                     args.target_task),
                       reward_entropy=reward_entropy(agent, validation,
                                                     args.target_task),
                       updates=0 if source is None else args.updates,
                       wall_seconds=time.monotonic() - started)
            results.append(row); save(out / 'results.json', results)
            print(json.dumps(row), flush=True)
    by = {(r['condition'], r['seed']): r for r in results}
    gates = []
    for seed in args.eval_seeds:
        steady, rnd = by[('steady', seed)], by[('random', seed)]
        er = steady['reward_entropy']['mean_entropy'] / max(rnd['reward_entropy']['mean_entropy'], 1e-12)
        gates.append(dict(seed=seed,
                          steady_loss_vs_random=1-steady['mpc_return']/max(rnd['mpc_return'],1e-8),
                          entropy_ratio_vs_random=er,
                          error_ratio_vs_random=steady['one_step_error']/max(rnd['one_step_error'],1e-12),
                          actor_delta=steady['actor_return']-rnd['actor_return'],
                          passed=(steady['mpc_return'] <= .85*rnd['mpc_return'] and er >= 1.10 and
                                  steady['one_step_error'] <= 1.05*rnd['one_step_error'] and
                                  abs(steady['actor_return']-rnd['actor_return']) <= .05*max(abs(rnd['actor_return']),1.0))))
    status=dict(status='completed', results=results, gates=gates,
                all_passed=all(g['passed'] for g in gates),
                decision='Short candidate screen only; do not scale unless both seeds and a second task family pass.',
                elapsed_seconds=time.monotonic()-started, api_requests=0)
    save(out / 'status.json', status); print(json.dumps(status), flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-task', default='cheetah-run')
    p.add_argument('--target-task', default='walker-run')
    p.add_argument('--source-seed', type=int, default=23)
    p.add_argument('--validation-seed', type=int, default=33)
    p.add_argument('--eval-seeds', nargs='+', type=int, default=[41,42])
    p.add_argument('--source-steps', type=int, default=300)
    p.add_argument('--eval-steps', type=int, default=300)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--learning-seed', type=int, default=31)
    p.add_argument('--output', required=True)
    main(p.parse_args())
