"""Bounded held-out native confirmation, including an action-multiset control."""
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import load_agent, episode, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt, frozen_digest
from screen_tdmpc_non_normal import collect, one_step_error


def save(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')


def replay_actions(agent, actions, seed):
    """Execute permuted legal commands; regenerate all observations/rewards."""
    set_seed(seed)
    task = agent.cfg.tasks.index('cheetah-run')
    env = make_one('cheetah-run', seed)
    observations, rewards = [pad_obs(env.reset(), agent.cfg)], []
    try:
        for action in actions:
            obs, reward, done, _ = env.step(action[:agent.cfg.action_dims[task]])
            observations.append(pad_obs(obs, agent.cfg))
            rewards.append(float(reward))
            if done:
                break
    finally:
        env.unwrapped.close()
    assert len(rewards) == len(actions)
    return dict(obs=torch.stack(observations), action=actions.clone(),
                reward=torch.tensor(rewards), task=task, seed=seed)


def main():
    out = Path('runs/tdmpc_cascade_heldout_walker_backwards_v1')
    out.mkdir(exist_ok=False)
    plan = dict(source_task='cheetah-run', source_seed=23, updates=300,
                learning_seed=31, source_steps=300, evaluation_steps=500,
                tasks=['walker-run-backwards'], evaluation_seeds=[31, 32],
                conditions=['clean', 'random', 'cascade', 'shuffled_cascade'],
                permutation_seed=933,
                scope='Exploratory follow-up fixed before execution. New source and evaluation seeds; full native 500-step episodes, eval_mode=False as in original experiments.',
                comparison='Shuffled cascade preserves the complete multiset of action vectors, not state coverage. Execute each sequence in a fresh real simulator; never reorder stored transitions.',
                gate='Cascade must lose >=15% return against BOTH random and shuffled controls at both held-out seeds; fixed random validation error <=1.05x random. Independent walker-backwards family must pass separately.',
                mechanism='No non-normal causality inferred from return or horizon sensitivity alone.',
                api_requests=0)
    save(out, 'plan.json', plan)
    start = time.monotonic()
    agent, provenance = load_agent('models/tdmpc2/mt30-48M.pt')
    save(out, 'provenance.json', provenance)
    cascade = collect(agent, 'cheetah-run', 23, 300, 'cascade')
    permutation = torch.randperm(300, generator=torch.Generator().manual_seed(933))
    validation = {task: collect(agent, task, 33, 300, 'random') for task in plan['tasks']}
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
        for task in plan['tasks']:
            error = one_step_error(agent, validation[task], task)
            for seed in plan['evaluation_seeds']:
                result, _ = episode(agent, task, seed, steps=500)
                row = dict(result, condition=condition, one_step_error=error,
                           planner_horizon=agent.cfg.horizon)
                rows.append(row)
                save(out, 'results.json', rows)
                print(json.dumps(row), flush=True)
    gates = {}
    for task in plan['tasks']:
        gates[task] = []
        for seed in plan['evaluation_seeds']:
            by = {r['condition']: r for r in rows if r['task']==task and r['seed']==seed}
            a = by['cascade']
            gates[task].append(dict(seed=seed,
                loss_vs_random=1-a['return']/by['random']['return'],
                loss_vs_shuffled=1-a['return']/by['shuffled_cascade']['return'],
                error_ratio=a['one_step_error']/by['random']['one_step_error'],
                passed=all(a['return'] <= .85*by[c]['return'] for c in ['random','shuffled_cascade']) and a['one_step_error']<=1.05*by['random']['one_step_error']))
    save(out, 'status.json', dict(status='completed', results=rows, gates=gates,
                                elapsed_seconds=time.monotonic()-start, api_requests=0))
    print(json.dumps(gates), flush=True)


if __name__ == '__main__':
    main()
