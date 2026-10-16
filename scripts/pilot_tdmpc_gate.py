"""Bounded clean capability check before any latent-model adaptation."""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from wm_security.tdmpc_native import load_agent, episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--tasks', nargs='+', default=['walker-walk', 'cheetah-run'])
    p.add_argument('--seeds', nargs='+', type=int, default=[11])
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--policies', nargs='+', choices=['mpc', 'prior', 'random'], default=['mpc', 'prior'])
    p.add_argument('--minimum-return', type=float, default=600)
    p.add_argument('--output', default='runs/tdmpc_clean_gate_v0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new output directory to preserve earlier runs.')
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    save('plan.json', dict(vars(args), scope='Published frozen multitask checkpoint; no learning or attack. Gate applies only to full500-step MPC episodes.'))
    started = time.monotonic(); results = []
    try:
        agent, provenance = load_agent(args.checkpoint)
        save('provenance.json', provenance)
        for task in args.tasks:
            for seed in args.seeds:
                for policy in args.policies:
                    print(json.dumps({'phase':'episode', 'task':task, 'seed':seed, 'policy':policy}), flush=True)
                    result, data = episode(agent, task, seed, policy, args.steps,
                                           progress=lambda x: print(json.dumps(x), flush=True))
                    torch.save(data, out / f'{task}_{seed}_{policy}.pt')
                    results.append(result);save('results.json', results)
                    print(json.dumps(result), flush=True)
        gate = bool(args.steps == 500 and 'mpc' in args.policies and
                    all(r['return'] >= args.minimum_return for r in results if r['policy'] == 'mpc'))
        save('status.json', {'status':'completed', 'clean_gate_passed':gate,
                            'results':results, 'wall_seconds':time.monotonic()-started,
                            'interpretation':'Clean runtime/capability screen only. No attack, online learning, shared-agent loss, or native continual-learning reproduction.'})
    except Exception as exc:
        save('status.json', {'status':'error', 'type':type(exc).__name__, 'error':str(exc),
                            'results':results, 'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
