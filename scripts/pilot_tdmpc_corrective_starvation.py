"""Short closed-loop screen for planner-induced corrective-data starvation.

The source contribution is a legal cascade or random trajectory.  After that
handoff, the victim collects truthful transitions with its own native MPC and
receives either only those transitions or an exploration-mixed continuation.
The proposed mechanism requires the former to preserve harm while the latter
repairs it; it is distinct from direct transition poisoning only if the
selection-induced data-support difference is causal.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt, frozen_digest
from screen_tdmpc_non_normal import collect


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def combine_data(a, b):
    return {k: torch.cat([a[k], b[k]]) if torch.is_tensor(a[k]) else a[k]
            for k in ['obs', 'action', 'reward']} | {'task': b['task'], 'seed': b['seed']}


def action_stats(data):
    x = data['action'].numpy()
    cov = np.cov(x[:, :data['action'].shape[1]].T) if len(x) > 1 else np.zeros((1, 1))
    return {'steps': int(len(x)), 'action_rms': float(np.sqrt(np.mean(x*x))),
            'action_abs_mean': float(np.mean(np.abs(x))),
            'action_cov_rank': int(np.linalg.matrix_rank(np.atleast_2d(cov), tol=1e-6)),
            'unique_rows_rounded': int(len(np.unique(np.round(x, 2), axis=0)))}


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan = dict(vars(args),
                hypothesis='A legal source contribution can steer the native planner away from corrective states; truthful victim MPC updates then preserve the loss, while exploration-mixed corrective data repairs it.',
                arms=['clean_mpc', 'random_source_mpc', 'cascade_source_mpc'],
                continuation=['mpc_only', 'mixed_random'],
                gate='Support-specific persistence requires cascade mpc_only harm >=15% versus clean, mixed_random recovery >=70% of loss, and cascade mpc_only action support lower than mixed_random; otherwise no promotion.',
                native_planner_unchanged=True, api_requests=0)
    save(out / 'plan.json', plan)
    rows = []
    for arm in ['clean_mpc', 'random_source_mpc', 'cascade_source_mpc']:
        base, provenance = load_agent(args.checkpoint)
        if arm == 'clean_mpc':
            source = None
        else:
            source = collect(base, args.source_task, args.source_seed, args.source_steps,
                             'random' if arm == 'random_source_mpc' else 'cascade')
            before = frozen_digest(base.model)
            adapt(base, source, args.source_updates, args.learning_seed, scope='dynamics')
            assert before == frozen_digest(base.model)
        pre, pre_data = episode(base, args.target_task, args.eval_seed, policy='mpc', steps=args.victim_steps)
        pre['arm'] = arm; pre['phase'] = 'pre_continuation'; pre['support'] = action_stats(pre_data)
        rows.append(pre); save(out / 'results.json', rows); print(json.dumps(pre), flush=True)
        for cont in ['mpc_only', 'mixed_random']:
            # Reset to the handoff checkpoint by repeating the source adaptation
            # so each continuation starts from identical model parameters.
            agent, _ = load_agent(args.checkpoint)
            if arm != 'clean_mpc':
                src = collect(agent, args.source_task, args.source_seed, args.source_steps,
                              'random' if arm == 'random_source_mpc' else 'cascade')
                adapt(agent, src, args.source_updates, args.learning_seed, scope='dynamics')
            _, mpc_data = episode(agent, args.target_task, args.eval_seed,
                                  policy='mpc', steps=args.victim_steps)
            if cont == 'mixed_random':
                _, rand_data = episode(agent, args.target_task, args.eval_seed + 100,
                                       policy='random', steps=args.victim_steps)
                update_data = combine_data(mpc_data, rand_data)
            else:
                update_data = mpc_data
            before = frozen_digest(agent.model)
            history = adapt(agent, update_data, args.continuation_updates,
                            args.learning_seed, scope='native_full')
            post, post_data = episode(agent, args.target_task, args.eval_seed,
                                      policy='mpc', steps=args.victim_steps)
            row = dict(arm=arm, phase=cont, pre_return=pre['return'],
                       post_return=post['return'], post_steps=post['steps'],
                       update_support=action_stats(update_data),
                       post_support=action_stats(post_data),
                       continuation_updates=args.continuation_updates,
                       update_digest_changed=before != frozen_digest(agent.model),
                       mean_total_loss=float(np.mean([h['total_loss'] for h in history])),
                       wall_seconds=time.monotonic() - started)
            rows.append(row); save(out / 'results.json', rows); print(json.dumps(row), flush=True)
    status = dict(status='completed', results=rows,
                  decision='Single-round closed-loop screen only; no broad persistence or attack claim without matched two-seed replication and a causal support intervention.',
                  elapsed_seconds=time.monotonic()-started, api_requests=0)
    save(out / 'status.json', status); print(json.dumps(status), flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-task', default='cheetah-run')
    p.add_argument('--target-task', default='walker-run')
    p.add_argument('--source-seed', type=int, default=23)
    p.add_argument('--eval-seed', type=int, default=41)
    p.add_argument('--source-steps', type=int, default=300)
    p.add_argument('--victim-steps', type=int, default=300)
    p.add_argument('--source-updates', type=int, default=80)
    p.add_argument('--continuation-updates', type=int, default=40)
    p.add_argument('--learning-seed', type=int, default=31)
    p.add_argument('--output', required=True)
    main(p.parse_args())
