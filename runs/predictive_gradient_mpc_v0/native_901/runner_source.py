"""Frozen stochastic dynamics with an explicitly researcher-defined gradient MPC.

Native DMControl task/reward; this is not the DMO actor or its training loop.
Sampler transforms use genuine forward operations, without custom backward.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch
from dm_control import suite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/DMO_code'))
from models.dyn_model import StochSSM


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def flatten(ts):
    return np.concatenate([np.asarray(v).ravel() for v in ts.observation.values()])


def reward(obs, action):
    """Exact smooth single-pole DMControl reward, in its bounded observation."""
    upright = (obs[..., 1] + 1) / 2
    centered = (1 + torch.exp(math.log(.1) * (obs[..., 0] / 2).square())) / 2
    small_velocity = (1 + torch.exp(math.log(.1) * (obs[..., 4] / 5).square())) / 2
    small_control = (5 - action[..., 0].square()) / 5
    return upright * centered * small_velocity * small_control


class Dynamics:
    def __init__(self, path, condition, strength, seed):
        self.condition, self.strength = condition, strength
        cfg = {'dyn_model_mlp': {'units': [512, 512], 'activation': 'silu', 'recurrent': False}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = StochSSM(5, 1, 5, cfg, False, device='cpu')
        self.model.load_state_dict(torch.load(path / 'clean_model.pt', map_location='cpu', weights_only=True))
        self.model.eval().requires_grad_(False)
        norm = json.loads((path / 'normalization.json').read_text())
        self.mean = torch.tensor(norm['input_mean'], dtype=torch.float32)
        self.std = torch.tensor(norm['input_std'], dtype=torch.float32)
        data = np.load(path / 'data_train.npz')
        x = torch.tensor(np.concatenate([data['obs'], data['action']], axis=1), dtype=torch.float32)
        with torch.no_grad():
            z = (x - self.mean) / self.std
            _, lv, _ = self.model(z[:, :5], z[:, 5:])
            self.reference_sigma = torch.exp(.5 * lv).median(dim=0).values.double()
        self.rng = torch.Generator().manual_seed(seed + 40000)
        self.predictive_transitions = 0
        self.grad_steps = 0

    def parameters(self, obs, action):
        z = (torch.cat([obs, action], dim=-1) - self.mean) / self.std
        delta, logvar, _ = self.model(z[..., :5], z[..., 5:])
        return obs + delta, (.5 * logvar).exp()

    def sample(self, obs, action):
        mu, sigma = self.parameters(obs, action)
        u = torch.rand(mu.shape, dtype=torch.float64, generator=self.rng)
        if self.condition == 'native':
            v = u
        else:
            phase = -self.strength * mu.double() / self.reference_sigma
            if self.condition == 'repair':
                # Trusted correction: disregard the unnecessary input-dependent
                # parametrization of noise. Forward values are exactly identical
                # to shifted at fixed input and fixed uniform variates.
                phase = phase.detach()
            v = torch.remainder(u + phase, 1.)
        if not bool(((v > 0) & (v < 1)).all()):
            raise FloatingPointError('Endpoint quantile; no clipping or resampling is hidden.')
        self.predictive_transitions += mu.shape[0]
        return mu + sigma * torch.special.ndtri(v).float()

    def rollout(self, obs, actions, particles):
        state = torch.tensor(obs, dtype=torch.float32).repeat(particles, 1)
        total = torch.zeros(particles)
        for action in actions:
            act = action.expand(particles, -1)
            state = self.sample(state, act)
            total = total + reward(state, act)
        return total.mean()


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    model_dir = Path(args.model)
    artifacts = [source, model_dir / 'clean_model.pt', model_dir / 'normalization.json',
                 ROOT / 'vendor/DMO_code/models/dyn_model.py']
    plan = dict(arguments=vars(args), hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
                scope='Native DMControl cartpole-balance with frozen DMO StochSSM component; researcher-defined Adam MPC, not native DMO control.',
                mechanism='Independent fresh uniforms per coordinate and transition, shifted by -strength*next_mean/reference_sigma modulo 1. Standard normal conditional law unchanged in exact arithmetic.',
                permission='Differentiable sampler source replacement; not weights-only, data-only, or action-only poisoning.',
                baseline_gate='First full native-sampler episode must score at least 900/1000 before evaluating attacks.',
                planner='Warm-started projected Adam on expected undiscounted native reward; execute first replan actions; no value oracle or custom backward.',
                api_requests=0, training_updates=0, gpu_work=False)
    save(out / 'plan.json', plan)
    dynamics = Dynamics(model_dir, args.condition, args.strength, args.seed)
    save(out / 'sampler.json', dict(reference_sigma=dynamics.reference_sigma.tolist(), fitted_on='Existing train data only'))
    env = suite.load('cartpole', 'balance', task_kwargs={'random': args.seed})
    ts = env.reset()
    obs = flatten(ts)
    warm = torch.zeros(args.horizon, 1)
    trace, planning = [], []
    reward_error = 0.
    start = time.monotonic()
    try:
        while len(trace) < args.steps and not ts.last():
            if time.monotonic() - start > args.max_seconds:
                raise TimeoutError('Explicit per-episode compute limit reached.')
            actions = warm.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([actions], lr=args.lr)
            values, grad_norms = [], []
            for _ in range(args.iterations):
                optimizer.zero_grad(set_to_none=True)
                value = dynamics.rollout(obs, actions, args.particles)
                (-value).backward()
                if not bool(torch.isfinite(actions.grad).all()):
                    raise FloatingPointError('Nonfinite action gradient.')
                grad_norms.append(float(actions.grad.norm()))
                torch.nn.utils.clip_grad_norm_([actions], args.grad_clip)
                optimizer.step()
                with torch.no_grad():
                    actions.clamp_(-1, 1)
                dynamics.grad_steps += 1
                values.append(float(value.detach()))
            action_plan = actions.detach()
            planning.append(dict(step=len(trace),values=values,grad_norms=grad_norms))
            for a in action_plan[:args.replan].numpy():
                old = obs
                ts = env.step(a.astype(env.action_spec().dtype))
                obs = flatten(ts)
                predicted_reward = float(reward(torch.tensor(obs), torch.tensor(a)))
                reward_error = max(reward_error, abs(predicted_reward - ts.reward))
                trace.append(dict(step=len(trace),obs=old.tolist(),action=a.tolist(),
                                  next_obs=obs.tolist(),reward=float(ts.reward),last=ts.last()))
                if ts.last() or len(trace) >= args.steps:
                    break
            warm = torch.cat([action_plan[args.replan:], torch.zeros(args.replan, 1)])
            if len(trace) % 100 == 0:
                print(json.dumps(dict(condition=args.condition,seed=args.seed,steps=len(trace),
                                      return_=sum(r['reward'] for r in trace),seconds=time.monotonic()-start)),flush=True)
        if reward_error > 1e-7:
            raise AssertionError(f'Reward parity failed: {reward_error}')
        result = dict(status='completed',condition=args.condition,seed=args.seed,
                      return_=sum(r['reward'] for r in trace),steps=len(trace),
                      full_native_episode=bool(ts.last()),reward_parity_max_error=reward_error,
                      predictive_transitions=dynamics.predictive_transitions,
                      action_optimizer_steps=dynamics.grad_steps,training_updates=0,
                      elapsed_seconds=time.monotonic()-start,api_requests=0,gpu_work=False)
    except Exception as exc:
        result = dict(status='failed',error=str(exc),condition=args.condition,seed=args.seed,
                      return_=sum(r['reward'] for r in trace),steps=len(trace),
                      predictive_transitions=dynamics.predictive_transitions,
                      action_optimizer_steps=dynamics.grad_steps,training_updates=0,
                      elapsed_seconds=time.monotonic()-start,api_requests=0,gpu_work=False)
        raise
    finally:
        env.close()
        save(out / 'trajectory.json', trace)
        save(out / 'planning.json', planning)
        if 'result' in locals():
            result['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(out / 'summary.json', result)
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='runs/dmo_jacobian_gate_v0')
    p.add_argument('--condition', choices=['native', 'shifted', 'repair'], required=True)
    p.add_argument('--seed', type=int, default=901)
    p.add_argument('--horizon', type=int, default=25)
    p.add_argument('--particles', type=int, default=10)
    p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--replan', type=int, default=5)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--strength', type=float, default=1.)
    p.add_argument('--grad-clip', type=float, default=100.)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--max-seconds', type=float, default=240.)
    main(p.parse_args())
