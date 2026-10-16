"""Short native screen for legal high-frequency-action temporal-gradient attack."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, episode
from pilot_tdmpc_adaptation import adapt
from pilot_tdmpc_action_search import source_error


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'models/tdmpc2/mt30-48M.pt'


def save(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + '\n')


def legal_sequence(agent, task, seed, steps, kind):
    cfg = agent.cfg
    tid = cfg.tasks.index(task)
    dim = cfg.action_dims[tid]
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), cfg)
    observations, actions, rewards = [obs], [], []
    gen = torch.Generator().manual_seed(seed + (100 if kind == 'white' else 0))
    try:
        for t in range(steps):
            action = torch.zeros(cfg.action_dim)
            if kind == 'dither':
                value = 1.0 if t % 2 == 0 else -1.0
                action[:dim] = value
            elif kind == 'white':
                action[:dim] = torch.rand(dim, generator=gen) * 2 - 1
            else:
                raise ValueError(kind)
            next_obs, reward, done, _ = env.step(action[:dim])
            observations.append(pad_obs(next_obs, cfg))
            actions.append(action)
            rewards.append(float(reward))
            if done:
                break
    finally:
        env.unwrapped.close()
    assert len(actions) == steps
    return {'obs': torch.stack(observations), 'action': torch.stack(actions),
            'reward': torch.tensor(rewards), 'task': tid, 'seed': seed,
            'kind': kind}


def main():
    out = ROOT / 'runs/tdmpc_temporal_gradient_dither_v0'
    out.mkdir(parents=True, exist_ok=False)
    source_task, victim_task = 'walker-walk-backwards', 'walker-walk'
    steps, updates, victim_seed = 150, 40, 12
    plan = {
        'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'source_task': source_task, 'victim_task': victim_task,
        'source_steps': steps, 'updates': updates, 'victim_seed': victim_seed,
        'conditions': ['clean', 'native_mpc_source', 'high_frequency_dither', 'equal_rms_white_noise'],
        'attacker_operation': 'Legal alternating +/-1 actuator commands in source task, authentic observations/rewards, ordinary native full updates.',
        'controls': 'White-noise legal commands with equal action range and native MPC source trajectory; clean victim-only branch.',
        'gate': 'Proceed only if dither causes >=100 lower native victim return than both white-noise and native-source controls while source one-step error remains finite.',
        'scope': 'One source/victim task pair, two-minute development screen; no long training or population claim.',
        'api_requests': 0, 'gpu_training': True,
    }
    save(out, 'plan.json', plan)
    rows = []
    started = time.monotonic()
    for condition in plan['conditions']:
        agent, provenance = load_agent(CHECKPOINT)
        if condition == 'clean':
            result, data = episode(agent, victim_task, victim_seed, 'mpc', 500)
            result['condition'] = condition
            rows.append(result)
            continue
        if condition == 'native_mpc_source':
            collection, source = episode(agent, source_task, 21, 'mpc', steps)
        elif condition == 'high_frequency_dither':
            source = legal_sequence(agent, source_task, 21, steps, 'dither')
        elif condition == 'equal_rms_white_noise':
            source = legal_sequence(agent, source_task, 21, steps, 'white')
        else:
            raise AssertionError(condition)
        torch.save(source, out / f'{condition}_source.pt')
        pre_error = source_error(agent, source)
        history = adapt(agent, source, updates, 31, 'native_full')
        save(out, f'{condition}_learning.json', {'pre_source_error': pre_error, 'history': history})
        result, data = episode(agent, victim_task, victim_seed, 'mpc', 500)
        result.update(condition=condition, pre_source_error=pre_error,
                      source_return=float(source['reward'].sum()),
                      final_total_loss=float(history[-1]['total_loss']))
        rows.append(result)
        torch.save(data, out / f'{condition}_victim.pt')
        print(json.dumps(result), flush=True)
    clean = next(r for r in rows if r['condition'] == 'clean')['return']
    by = {r['condition']: r for r in rows}
    dither_return = by['high_frequency_dither']['return']
    comparison = {
        'clean_return': clean,
        'dither_return': dither_return,
        'dither_drop_vs_clean': clean - dither_return,
        'dither_drop_vs_white': by['equal_rms_white_noise']['return'] - dither_return,
        'dither_drop_vs_native_source': by['native_mpc_source']['return'] - dither_return,
        'gate_passed': (clean - dither_return >= 100 and
                        by['equal_rms_white_noise']['return'] - dither_return >= 100 and
                        by['native_mpc_source']['return'] - dither_return >= 100),
    }
    summary = {
        'status': 'completed', 'results': rows, 'comparison': comparison,
        'decision': 'Run native confirmation only if the fixed dither gate passes. Otherwise retire as ordinary legal action-distribution adaptation.',
        'elapsed_seconds': time.monotonic() - started,
        'native_environment_steps': sum(r['steps'] for r in rows),
        'api_requests': 0, 'gpu_training': True,
    }
    save(out, 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
