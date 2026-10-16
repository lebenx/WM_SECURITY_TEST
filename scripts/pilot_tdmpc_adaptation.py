"""Short real-data adaptation screen, not a learned attack or native online run.

The default isolates shared dynamics with the upstream three-step consistency
objective; other components remain frozen. The native_full option calls the
upstream joint update function including encoder, reward/value, and policy.
The published multitask checkpoint is offline; this continuation is an explicit
researcher-defined deployment experiment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from wm_security.tdmpc_native import load_agent, episode, set_seed


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('_dynamics.'):
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def adapt(agent, data, updates, seed, scope='dynamics'):
    """Native complete update or an isolated dynamics-loss intervention."""
    set_seed(seed)
    cfg, model = agent.cfg, agent.model
    if scope == 'native_full':
        # A prior dynamics-only probe may have frozen the remaining modules.
        # Restore the upstream full-update contract before calling _update.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        cfg.batch_size = 32
        observations = data['obs'].to(agent.device)
        actions = data['action'].to(agent.device)
        rewards = data['reward'].to(agent.device)
        tasks = torch.full((cfg.batch_size,), data['task'], device=agent.device, dtype=torch.long)
        history = []
        for update in range(updates):
            starts = torch.randint(len(actions) - cfg.horizon + 1, (cfg.batch_size,), device=agent.device)
            obs = torch.stack([observations[starts+t] for t in range(cfg.horizon+1)])
            action = torch.stack([actions[starts+t] for t in range(cfg.horizon)])
            reward = torch.stack([rewards[starts+t] for t in range(cfg.horizon)]).unsqueeze(-1)
            terminated = torch.zeros_like(reward)
            info = agent._update(obs, action, reward, terminated, tasks)
            history.append({'update':update+1, **{k:float(v) for k,v in info.items()}})
        return history
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model._dynamics.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model._dynamics.parameters(), lr=cfg.lr)
    observations = data['obs'].to(agent.device)
    actions = data['action'].to(agent.device)
    task = data['task']
    model.eval()
    # Fixed encoder targets prevent moving-representation confounds.
    with torch.no_grad():
        encoded = model.encode(observations, task)
    history = []
    for update in range(updates):
        starts = torch.randint(len(actions) - cfg.horizon + 1, (32,), device=agent.device)
        z = encoded[starts]
        loss = torch.zeros((), device=agent.device)
        for step in range(cfg.horizon):
            z = model.next(z, actions[starts + step], task)
            loss = loss + cfg.rho**step * F.mse_loss(z, encoded[starts + step + 1])
        loss = loss / cfg.horizon * cfg.consistency_coef
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model._dynamics.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        history.append({'update':update + 1, 'weighted_consistency_loss':float(loss.detach()), 'grad_norm':float(grad)})
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-task', default='walker-walk-backwards')
    p.add_argument('--targets', nargs='+', default=['walker-walk-backwards', 'walker-walk'])
    p.add_argument('--source-seed', type=int, default=21)
    p.add_argument('--eval-seed', type=int, default=11)
    p.add_argument('--actions', type=int, default=300)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--update-scope', choices=['dynamics', 'native_full'], default='dynamics')
    p.add_argument('--output', default='runs/tdmpc_adaptation_screen_v0')
    args = p.parse_args()
    if not 1 <= args.updates <= 100 or not 3 <= args.actions <= 500:
        raise ValueError('This entry point is restricted to short pilots.')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new output directory to preserve earlier runs.')
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, default=str)+'\n')
    plan = dict(vars(args), source_policies=['mpc', 'random'],
                optimizer='Adam, upstream lr, batch32; native full update or dynamics-only consistency ablation as configured',
                scope='Researcher-defined continual adaptation of an offline pretrained model. No observation edits, learned adversarial policy, or attack claim.')
    save('plan.json', plan)
    results = []
    try:
        agent, provenance = load_agent(args.checkpoint)
        save('provenance.json', provenance)
        original = {k:v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                    for k,v in agent.model.state_dict().items()}
        for target in args.targets:
            r, d = episode(agent, target, args.eval_seed)
            r['condition'] = 'clean';results.append(r)
            torch.save(d, out/f'clean_{target}.pt');save('results.json', results)
            print(json.dumps(r), flush=True)
        if any(r['return'] < 600 for r in results):
            save('status.json', {'status':'clean_gate_failed', 'results':results});return
        for policy in plan['source_policies']:
            agent.model.load_state_dict(original)
            collection, data = episode(agent, args.source_task, args.source_seed, policy, args.actions)
            save(f'{policy}_collection.json', collection);torch.save(data, out/f'{policy}_data.pt')
            if args.update_scope == 'native_full':
                # Fresh optimizer, target networks, and running scale per arm.
                agent, _ = load_agent(args.checkpoint)
            before = frozen_digest(agent.model)
            started = time.monotonic()
            history = adapt(agent, data, args.updates, seed=31, scope=args.update_scope)
            elapsed = time.monotonic() - started
            after = frozen_digest(agent.model)
            if args.update_scope == 'dynamics' and after != before:
                raise AssertionError('A frozen model component changed during dynamics adaptation.')
            save(f'{policy}_adaptation.json', {'updates':history, 'wall_seconds':elapsed,
                                              'update_scope':args.update_scope,
                                              'non_dynamics_sha256_before':before, 'non_dynamics_sha256_after':after})
            torch.save(agent.model._dynamics.state_dict(), out/f'{policy}_dynamics.pt')
            for target in args.targets:
                r, d = episode(agent, target, args.eval_seed)
                r['condition'] = policy;results.append(r)
                torch.save(d, out/f'{policy}_{target}_evaluation.pt');save('results.json', results)
                print(json.dumps(r), flush=True)
        save('status.json', {'status':'completed', 'results':results,
                            'interpretation':'One seed, bounded adaptation with recorded update scope. Random and ordinary real experience, not a targeted attack. Cross-task return changes are exploratory and do not establish novelty, population effects, or persistent collective harm.'})
    except Exception as exc:
        save('status.json', {'status':'error','error':str(exc),'traceback':traceback.format_exc(),'results':results})
        raise


if __name__ == '__main__':
    main()
