"""Bounded inference-only screen of a researcher-defined shared planner service.

The upstream single-agent planner is not advertised as tenant-safe. This pilot
shares its instance deliberately; it is not evidence of a native deployed
multi-user vulnerability or of experience poisoning. Attacker inputs are
unedited observations from an existing real simulator trajectory. Only native
act() calls alter the shared warm start in the interference arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(list(model.named_parameters()) + list(model.named_buffers())):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_episode(agent, source, task, seed, condition, progress):
    cfg = agent.cfg
    task_id = cfg.tasks.index(task)
    set_seed(seed)
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), cfg)
    observations, actions, rewards, warm_start_changes = [obs.clone()], [], [], []
    started = time.monotonic()
    try:
        for step in range(500):
            if condition != 'clean':
                previous = agent._prev_mean.clone()
                index = step % len(source['action'])
                # Isolate random-stream consumption from planner-state effects.
                # The attacker does not see victim observations or rewards.
                with torch.random.fork_rng(devices=[agent.device.index or 0]):
                    torch.manual_seed(100000 + step)
                    torch.cuda.manual_seed_all(100000 + step)
                    agent.act(source['obs'][index], t0=index == 0,
                              task=int(source['task']), eval_mode=False)
                warm_start_changes.append(float((agent._prev_mean - previous).norm()))
                if condition == 'restore_victim_warm_start':
                    agent._prev_mean.copy_(previous)
            action = agent.act(obs, t0=step == 0, task=task_id, eval_mode=False)
            next_obs, reward, done, _ = env.step(action[:cfg.action_dims[task_id]])
            obs = pad_obs(next_obs, cfg)
            observations.append(obs.clone())
            actions.append(action.clone())
            rewards.append(float(reward))
            if (step + 1) % 100 == 0:
                progress({'condition': condition, 'step': step + 1, 'return': sum(rewards)})
            if done:
                break
    finally:
        env.unwrapped.close()
    return {
        'condition': condition, 'task': task, 'seed': seed,
        'steps': len(actions), 'full_episode': len(actions) == 500,
        'return': sum(rewards), 'attacker_queries': len(warm_start_changes),
        'mean_query_warm_start_change_l2': (
            sum(warm_start_changes) / len(warm_start_changes) if warm_start_changes else 0),
        'wall_seconds': time.monotonic() - started,
    }, {
        'obs': torch.stack(observations), 'action': torch.stack(actions),
        'reward': torch.tensor(rewards),
        'query_warm_start_change_l2': torch.tensor(warm_start_changes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    parser.add_argument('--source', default='runs/tdmpc_adaptation_screen_v1/mpc_data.pt')
    parser.add_argument('--target', default='walker-walk')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', default='runs/tdmpc_query_interference_v0')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')

    plan = dict(
        vars(args),
        conditions=['clean', 'shared_planner', 'restore_victim_warm_start'],
        scope='Researcher-defined sharing of an upstream single-agent planner instance; no native multi-tenant deployment claim.',
        attacker='One native planning query per victim action, cycling unedited observations from a recorded real source episode; no victim feedback, observation editing, state-tensor writes, or learning updates.',
        intervention='Restore the victim warm start after the exact same attacker-query schedule.',
        randomness='Fork and restore PyTorch CPU/CUDA RNG around attacker queries; fixed attacker seed100000+step. Native victim sampling and optimization budgets unchanged.',
        clean_gate_return=800,
        effect_gate_return_drop=100,
        confirmation='Only a development screen; a positive result would still require independent seeds, budget-matched benign/random query baselines, a second native mechanism, and a novelty check.',
        api_requests=0,
        training_steps=0,
    )
    save('plan.json', plan)
    results, traces = [], {}
    try:
        source = torch.load(args.source, map_location='cpu', weights_only=True)
        agent, provenance = load_agent(args.checkpoint)
        provenance['source_task'] = agent.cfg.tasks[int(source['task'])]
        provenance['source_actions'] = len(source['action'])
        provenance['source_sha256'] = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        save('provenance.json', provenance)
        before = model_digest(agent.model)
        for condition in plan['conditions']:
            agent._prev_mean.zero_()
            result, trace = run_episode(
                agent, source, args.target, args.seed, condition,
                lambda record: print(json.dumps(record), flush=True))
            results.append(result)
            traces[condition] = trace
            torch.save(trace, out / (condition + '.pt'))
            save('results.json', results)
            if condition == 'clean' and result['return'] < plan['clean_gate_return']:
                save('status.json', {'status': 'clean_gate_failed', 'results': results})
                return
        after = model_digest(agent.model)
        if before != after:
            raise AssertionError('The frozen world model or policy changed.')
        restored = all(torch.equal(traces['clean'][field],
                                   traces['restore_victim_warm_start'][field])
                       for field in ['obs', 'action', 'reward'])
        if not restored:
            raise AssertionError('Warm-start restoration failed to reproduce the clean trajectory.')
        drop = results[0]['return'] - results[1]['return']
        status = {
            'status': 'completed', 'results': results,
            'shared_planner_return_drop': drop,
            'effect_gate_passed': drop >= plan['effect_gate_return_drop'],
            'restoration_exactly_reproduces_clean': restored,
            'model_and_policy_sha256_before': before,
            'model_and_policy_sha256_after': after,
            'interpretation': 'Single development episode under explicit unsafe sharing assumption. State changes alone are not attack success. No learning, adversarial optimization, native multi-tenant deployment, generality or novelty claim.',
        }
        save('status.json', status)
        print(json.dumps({k: v for k, v in status.items() if k != 'results'}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'error', 'error': str(exc),
                             'traceback': traceback.format_exc(), 'results': results})
        raise


if __name__ == '__main__':
    main()
