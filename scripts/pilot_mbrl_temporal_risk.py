"""Bounded risk-sensitive check for temporal-copula imagination changes.

This keeps the native MBRL-Lib CEM optimizer and GaussianMLP, but scores each
candidate by the lower-tail mean across model particles instead of the particle
mean.  The only intervention is the TemporalCoupler's cross-time residual
coupling; no model weights or environment dynamics are changed.
"""
from __future__ import annotations

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
from wm_security.temporal_coupling import TemporalCoupler
from wm_security.epistemic_collapse import EpistemicCollapseCoupler
from pilot_mbrl_coupling import make_model, model_hash, save


@torch.no_grad()
def evaluate_particles(model_env, actions, initial_state, num_particles):
    """Return B x P accumulated particle returns using native ModelEnv steps."""
    assert len(actions.shape) == 3
    population, horizon, _ = actions.shape
    initial_obs_batch = np.tile(
        initial_state, (num_particles * population,) + (1,) * initial_state.ndim
    ).astype(np.float32)
    state = model_env.reset(initial_obs_batch, return_as_np=False)
    total = torch.zeros(num_particles * population, 1).to(model_env.device)
    terminated = torch.zeros(num_particles * population, 1, dtype=torch.bool).to(model_env.device)
    for time_step in range(horizon):
        action = torch.repeat_interleave(actions[:, time_step, :], num_particles, dim=0)
        _, rewards, dones, state = model_env.step(action, state, sample=True)
        rewards[terminated] = 0
        terminated |= dones
        total += rewards
    return total.reshape(population, num_particles)


def evaluate(args, out):
    model = make_model(); model.load(Path(args.model)); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    digest = model_hash(model)
    rows = []
    started = time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        env = CartPoleEnv(); obs, _ = env.reset(seed)
        model_env = ModelEnv(
            env, model, termination_fns.cartpole, reward_fns.cartpole,
            generator=torch.Generator().manual_seed(seed + 10000),
        )
        cfg = OmegaConf.create(dict(
            _target_='mbrl.planning.CEMOptimizer', num_iterations=3,
            elite_ratio=.1, population_size=100, alpha=.1, device='cpu',
            return_mean_elites=False,
        ))
        agent = TrajectoryOptimizerAgent(cfg, [-1.], [1.], planning_horizon=15, replan_freq=1)
        coupler = None
        if args.condition == 'persistent':
            coupler = TemporalCoupler(model_env, 'persistent'); coupler.install()
        elif args.condition == 'collapse':
            coupler = EpistemicCollapseCoupler(model_env, args.theta_threshold); coupler.install()

        def objective(initial, action_sequences):
            particles = evaluate_particles(model_env, action_sequences, initial, args.particles)
            if args.objective == 'mean':
                return particles.mean(dim=1)
            k = max(1, int(np.ceil(args.alpha * args.particles)))
            return particles.topk(k, dim=1, largest=False).values.mean(dim=1)

        agent.set_trajectory_eval_fn(objective)
        total = 0.; trajectory = []; t0 = time.monotonic()
        try:
            for step in range(200):
                action = agent.act(obs)
                nxt, reward, done, truncated, _ = env.step(action)
                trajectory.append(dict(step=step, obs=obs.tolist(), action=action.tolist(),
                                       next_obs=nxt.tolist(), reward=float(reward), terminated=bool(done)))
                total += float(reward); obs = nxt
                if done or truncated: break
        finally:
            if coupler: coupler.restore()
            env.close()
        assert model_hash(model) == digest
        row = dict(condition=args.condition, objective=args.objective, alpha=args.alpha,
                   particles=args.particles, seed=seed, return_=total,
                   steps=len(trajectory), survived_200=len(trajectory) == 200 and not trajectory[-1]['terminated'],
                   seconds=time.monotonic() - t0, model_hash_unchanged=True)
        if coupler:
            if args.condition == 'persistent':
                row.update(temporal_calls=coupler.total_steps, marginal_rows=coupler.marginal_rows,
                           used_residual_mean=coupler.used_sum / max(1, coupler.used_count),
                           used_residual_std=float(max(0., coupler.used_sq_sum / max(1, coupler.used_count) -
                                                       (coupler.used_sum / max(1, coupler.used_count)) ** 2) ** .5),
                           adjacent_residual_product_mean=coupler.autocorrelation_sum /
                           max(1, coupler.autocorrelation_count))
            else:
                row.update(triggered_fraction=coupler.triggered_fraction,
                           triggered_ensemble_spread=coupler.triggered_ensemble_spread)
        rows.append(row)
        save(out / ('episode_' + str(seed) + '.json'), dict(summary=row, trajectory=trajectory))
        print(json.dumps(row), flush=True)
    result = dict(status='completed', condition=args.condition, objective=args.objective,
                  alpha=args.alpha, particles=args.particles, rows=rows, model_hash=digest,
                  mean_return=float(np.mean([r['return_'] for r in rows])),
                  native_environment_steps=sum(r['steps'] for r in rows),
                  elapsed_seconds=time.monotonic() - started, training_updates=0,
                  api_requests=0, gpu_work=False,
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out / 'summary.json', result); print(json.dumps(result), flush=True)


def main(args):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    files = ['scripts/pilot_mbrl_temporal_risk.py', 'src/wm_security/temporal_coupling.py',
             'src/wm_security/epistemic_collapse.py',
             'scripts/pilot_mbrl_coupling.py']
    save(out / 'plan.json', dict(arguments=vars(args), hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
                                  model='Frozen native five-member MBRL-Lib GaussianMLP from mbrl_coupling_cartpole_v0.',
                                  objective='Mean or lower-tail particle return.',
                                  intervention=('Triggered ensemble-mean replacement; reduces epistemic spread and changes predictive laws.'
                                                if args.condition == 'collapse' else
                                                'Persistent residuals preserve unconditional residual marginals, not conditional transition laws.'
                                                if args.condition == 'persistent' else 'Unmodified model sampling.'),
                                  promotion_gate='For this two-seed screen, both native returns >=180 and both paired attacked returns <=80% of native. Passing justifies independent confirmation, not acceptance of a claim.',
                                  scope='Bounded two or more seed gate; no publication-scale claim.', training_updates=0,
                                  api_requests=0, gpu_work=False))
    evaluate(args, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--model', required=True)
    p.add_argument('--condition', choices=['native', 'persistent', 'collapse'], required=True)
    p.add_argument('--objective', choices=['mean', 'cvar'], default='cvar')
    p.add_argument('--alpha', type=float, default=.2); p.add_argument('--particles', type=int, default=20)
    p.add_argument('--theta-threshold', type=float, default=.08)
    p.add_argument('--seeds', nargs='+', type=int, default=[810, 811])
    args = p.parse_args()
    if not 0 < args.alpha <= 1 or args.particles < 1:
        p.error('alpha must be in (0, 1] and particles must be positive')
    main(args)
