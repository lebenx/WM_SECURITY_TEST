"""Held-out native cascade gate on an independent hopper-hop task family."""
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt, frozen_digest
from screen_tdmpc_non_normal import collect, one_step_error
from confirm_tdmpc_cascade import replay_actions


def save(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    out = Path('runs/tdmpc_cascade_heldout_hopper_v1')
    out.mkdir(exist_ok=False)
    plan = dict(source_task='cheetah-run', source_seed=23, updates=300,
                source_steps=300, evaluation_steps=500, task='hopper-hop',
                evaluation_seeds=[31, 32],
                conditions=['clean', 'random', 'cascade', 'shuffled_cascade'],
                permutation_seed=933,
                scope='Independent held-out positive-family gate using the same native TD-MPC2 protocol as the walker-run confirmation.',
                gate='Cascade must lose >=15% return against BOTH random and shuffled controls at both seeds; fixed validation error <=1.05x random.',
                api_requests=0)
    save(out, 'plan.json', plan)
    start = time.monotonic()
    agent, provenance = load_agent('models/tdmpc2/mt30-48M.pt')
    save(out, 'provenance.json', provenance)
    cascade = collect(agent, 'cheetah-run', 23, 300, 'cascade')
    permutation = torch.randperm(300, generator=torch.Generator().manual_seed(933))
    validation = collect(agent, 'hopper-hop', 33, 300, 'random')
    rows = []
    for condition in plan['conditions']:
        if condition != 'clean':
            agent, _ = load_agent('models/tdmpc2/mt30-48M.pt')
            if condition == 'cascade':
                source = cascade
            elif condition == 'shuffled_cascade':
                source = replay_actions(agent, cascade['action'][permutation], 23)
                torch.testing.assert_close(source['action'][torch.argsort(permutation)], cascade['action'], rtol=0, atol=0)
            else:
                source = collect(agent, 'cheetah-run', 23, 300, 'random')
            torch.save(source, out / f'{condition}_source.pt')
            before = frozen_digest(agent.model)
            history = adapt(agent, source, 300, 31, 'dynamics')
            assert before == frozen_digest(agent.model)
            save(out, f'{condition}_learning.json', history)
        error = one_step_error(agent, validation, 'hopper-hop')
        for seed in plan['evaluation_seeds']:
            result, _ = episode(agent, 'hopper-hop', seed, steps=500)
            row = dict(result, condition=condition, one_step_error=error,
                       planner_horizon=agent.cfg.horizon)
            rows.append(row)
            save(out, 'results.json', rows)
            print(json.dumps(row), flush=True)
    by_seed = {}
    for seed in plan['evaluation_seeds']:
        by = {r['condition']: r for r in rows if r['seed'] == seed}
        a = by['cascade']
        by_seed[str(seed)] = dict(
            loss_vs_random=1 - a['return'] / by['random']['return'],
            loss_vs_shuffled=1 - a['return'] / by['shuffled_cascade']['return'],
            error_ratio=a['one_step_error'] / by['random']['one_step_error'],
            passed=all(a['return'] <= .85 * by[c]['return']
                       for c in ['random', 'shuffled_cascade'])
                   and a['one_step_error'] <= 1.05 * by['random']['one_step_error'])
    status = dict(status='completed', results=rows, gates=by_seed,
                  elapsed_seconds=time.monotonic() - start, api_requests=0,
                  decision='Independent positive-family gate only; no broad mechanism is inferred from this screen.')
    save(out, 'status.json', status)
    print(json.dumps(by_seed), flush=True)


if __name__ == '__main__':
    main()
