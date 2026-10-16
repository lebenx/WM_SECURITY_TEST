"""Native MBRL-Lib CartPole pilot for third-order copula coupling."""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns, termination_fns
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from pilot_mbrl_coupling import make_model, model_hash, save
from wm_security.parity_coupling import ParitySampler


def evaluate(args, out):
    model = make_model(); model.load(Path(args.model)); model.eval()
    before = model_hash(model)
    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        env = CartPoleEnv(); obs, _ = env.reset(seed)
        model_env = ModelEnv(env, model, termination_fns.cartpole, reward_fns.cartpole,
                             generator=torch.Generator().manual_seed(seed + 10000))
        cfg = OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer', num_iterations=3,
                                    elite_ratio=.1, population_size=args.population, alpha=.1,
                                    device='cpu', return_mean_elites=False))
        agent = TrajectoryOptimizerAgent(cfg, [-1.], [1.], planning_horizon=args.horizon, replan_freq=1)
        sampler = ParitySampler(model_env, args.particles, args.condition, seed + 30000)
        agent.set_trajectory_eval_fn(sampler.evaluate)
        total = 0.; steps = 0; started = time.monotonic()
        try:
            for steps in range(args.max_steps):
                action = agent.act(obs)
                nxt, reward, done, truncated, _ = env.step(action)
                total += float(reward); obs = nxt
                if done or truncated: break
        finally:
            env.close()
        assert model_hash(model) == before
        row = dict(condition=args.condition, seed=seed, return_=total, steps=steps + 1,
                   model_hash_unchanged=True, evaluations=sampler.calls,
                   predictive_transitions=sampler.predictive_transitions,
                   triples=sampler.triples, attack_orientations=sampler.attack_orientations,
                   seconds=time.monotonic() - started)
        rows.append(row); save(out / f'episode_{seed}.json', row); print(json.dumps(row), flush=True)
    result = dict(status='completed', condition=args.condition, rows=rows,
                  mean_return=float(np.mean([r['return_'] for r in rows])),
                  model_hash=before, api_requests=0, gpu_work=False,
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out / 'summary.json', result); print(json.dumps(result), flush=True)


def main(args):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    files = ['scripts/pilot_mbrl_parity_coupling.py', 'src/wm_security/parity_coupling.py']
    save(out / 'plan.json', dict(arguments=vars(args),
         hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
         operation='Two-row native forecast banks coupled over triples by even/odd parity; pairwise index law is exact for every realized bank.',
         controls='independent two-row sampling and parity-control orientation that maximizes the same local CEM score.',
         gate='Only promote if parity attack harms native CartPole against both independent and parity-control conditions while exact pairwise checks pass.',
         scope='Short native task pilot; no broad claim from this run.', api_requests=0, gpu_work=False))
    evaluate(args, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--model', required=True)
    p.add_argument('--condition', choices=['independent', 'parity_attack', 'parity_control'], required=True)
    p.add_argument('--particles', type=int, default=10); p.add_argument('--population', type=int, default=100)
    p.add_argument('--horizon', type=int, default=15); p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--seeds', nargs='+', type=int, default=[704, 705]); main(p.parse_args())
