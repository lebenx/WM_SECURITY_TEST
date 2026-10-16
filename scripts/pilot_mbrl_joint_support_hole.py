"""Native MBRL-Lib gate for conditional action-support holes.

This keeps the MBRL-Lib GaussianMLP, ModelEnv, ModelTrainer and CEM planner,
but uses a tiny contextual point-mass environment whose context is observed
and whose dynamics contain a real context-action interaction.  The attacker
chooses only legal actions and preserves context/action marginals; it removes
joint context-action cells.  The run is deliberately short and is a gate for
whether a native carrier warrants broader tasks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from mbrl.env import termination_fns
from mbrl.models import GaussianMLP, ModelEnv, ModelTrainer, OneDTransitionRewardModel
from mbrl.planning import TrajectoryOptimizerAgent
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator


class ContextualPointEnv(gym.Env):
    def __init__(self, seed=0, horizon=25):
        self.observation_space = spaces.Box(
            low=np.array([-2., -1., -1., -1.], dtype=np.float32),
            high=np.array([2., 1., 1., 1.], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1.], dtype=np.float32),
                                       np.array([1.], dtype=np.float32), dtype=np.float32)
        self.horizon = horizon
        self.np_random = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        x = self.np_random.uniform(-.05, .05)
        p, q = self.np_random.choice([-1., 1.], size=2)
        self.state = np.array([x, p, q, 1.], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1., 1.))
        x, p, q, t = [float(v) for v in self.state]
        # The phase t alternates every step. The attacker uses sign(a)=p*q*t;
        # each context cell sees one action sign while each interaction term
        # has zero mean over the independent signs and alternating phase.
        x_next = .92 * x + a + .25 * p * a + .25 * q * a + .25 * t * a + .45 * q * (a * a)
        self.state = np.array([x_next, p, q, -t], dtype=np.float32)
        self.steps += 1
        target = .25
        reward = -abs(x_next - target) - .01 * a * a
        terminated = bool(abs(x_next) > 2.0)
        truncated = self.steps >= self.horizon
        return self.state.copy(), float(reward), terminated, truncated, {}


def truth(obs, action):
    x, p, q, t = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3]
    a = action[..., 0]
    return np.stack([.92 * x + a + .25 * p * a + .25 * q * a + .25 * t * a + .45 * q * (a * a),
                     p, q, -t], axis=-1).astype(np.float32)


def make_model():
    base = GaussianMLP(5, 4, 'cpu', num_layers=2, ensemble_size=5, hid_size=64,
                       deterministic=False, propagation_method='fixed_model',
                       activation_fn_cfg={'_target_': 'torch.nn.LeakyReLU',
                                          'negative_slope': .01})
    return OneDTransitionRewardModel(base, target_is_delta=False, normalize=True,
                                     learned_rewards=False, obs_process_fn=None,
                                     no_delta_list=[])


def collect(mode, seed, transitions):
    env = ContextualPointEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    rng = np.random.default_rng(seed + 100)
    for step in range(transitions):
        if mode == 'clean' or mode == 'permuted_control':
            action = np.array([rng.uniform(-.65, .65)], dtype=np.float32)
        elif mode == 'support_hole':
            # p, q and t are observed; actions remain balanced globally but p*q*t*a is always
            # positive. Every transition and reward comes from the environment.
            action = np.array([float(obs[1] * obs[2] * obs[3]) * rng.uniform(0., .65)], dtype=np.float32)
        else:
            raise ValueError(mode)
        nxt, reward, term, trunc, _ = env.step(action)
        rows.append((obs.copy(), action.copy(), nxt.copy(), reward, term, trunc))
        obs = nxt
        if term or trunc:
            obs, _ = env.reset(seed=seed + step + 1)
    env.close()
    data = [np.asarray(x, dtype=bool if j >= 4 else np.float32)
            for j, x in enumerate(zip(*rows))]
    return TransitionBatch(*data)


def train_model(batch, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(batch))
    split = int(.85 * len(batch))
    tr, va = batch[idx[:split]], batch[idx[split:]]
    model = make_model()
    model.update_normalizer(tr)
    trainer = ModelTrainer(model, optim_lr=2e-3, weight_decay=1e-5)
    training = BootstrapIterator(tr, 128, 5, shuffle_each_epoch=True,
                                 permute_indices=False, rng=rng)
    validation = TransitionIterator(va, 256)
    trainer.train(training, validation, num_epochs=epochs, patience=8, silent=True)
    return model, tr, va, float(trainer.evaluate(validation).mean())


def evaluate_errors(model, seed, n=3000):
    rng = np.random.default_rng(seed + 500)
    x = rng.uniform(-.5, .5, n).astype(np.float32)
    p = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    q = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    t = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    mag = rng.uniform(0., .65, n).astype(np.float32)
    sign = np.where(rng.random(n) < .5, 1., -1.).astype(np.float32)
    a = (p * q * t * mag * sign).astype(np.float32)
    obs = np.column_stack([x, p, q, t]).astype(np.float32)
    act = a[:, None]
    actual = truth(obs, act)
    inp = torch.from_numpy(np.column_stack([obs, act]))
    with torch.no_grad():
        pred = model.model(inp, use_propagation=False)[0].mean(0).detach().cpu().numpy()
    observed = p * q * t * a >= 0
    return {
        'one_step_mse_observed': float(np.mean((pred[observed] - actual[observed]) ** 2)),
        'one_step_mse_missing': float(np.mean((pred[~observed] - actual[~observed]) ** 2)),
        'missing_observed_ratio': float(np.mean((pred[~observed] - actual[~observed]) ** 2) /
                                        max(np.mean((pred[observed] - actual[observed]) ** 2), 1e-12)),
        'x_mse_observed': float(np.mean((pred[observed, 0] - actual[observed, 0]) ** 2)),
        'x_mse_missing': float(np.mean((pred[~observed, 0] - actual[~observed, 0]) ** 2)),
        'x_missing_observed_ratio': float(np.mean((pred[~observed, 0] - actual[~observed, 0]) ** 2) /
                                          max(np.mean((pred[observed, 0] - actual[observed, 0]) ** 2), 1e-12)),
        'state_abs_mean': float(np.mean(np.abs(x))),
        'action_mean': float(np.mean(a)),
        'action_abs_mean': float(np.mean(np.abs(a))),
        'observed_fraction': float(np.mean(observed)),
    }


def rollout_cem(model, seed, episodes=2, steps=20):
    env = ContextualPointEnv(seed=seed, horizon=steps)
    cfg = OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',
                                num_iterations=3, elite_ratio=.1,
                                population_size=96, alpha=.1,
                                device='cpu', return_mean_elites=True,
                                clipped_normal=False))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep * 100)
        model_env = ModelEnv(env, model, termination_fns.no_termination,
                             reward_fn=lambda act, nxt: (
                                 -torch.abs(nxt[:, :1] - .25) - .01 * act.square().sum(1, keepdim=True)))
        agent = TrajectoryOptimizerAgent(cfg, [-1.], [1.], planning_horizon=5, replan_freq=1)
        agent.set_trajectory_eval_fn(
            lambda initial_obs, action_sequences: model_env.evaluate_action_sequences(
                action_sequences, initial_obs, num_particles=5))
        total = 0.
        for step in range(steps):
            action = agent.act(obs)
            nxt, reward, term, trunc, _ = env.step(action)
            total += reward
            obs = nxt
            if term or trunc:
                break
        rows.append({'episode': ep, 'return': float(total), 'p': float(obs[1]),
                     'q': float(obs[2]), 't': float(obs[3])})
    env.close()
    return {'return_mean': float(np.mean([r['return'] for r in rows])), 'episodes': rows}


def main(args):
    started = time.monotonic()
    rows = []
    for seed in args.seeds:
        for mode in ['clean', 'support_hole', 'permuted_control']:
            data = collect(mode, seed, args.transitions)
            model, tr, va, val = train_model(data, seed + {'clean': 0, 'support_hole': 100, 'permuted_control': 200}[mode], args.epochs)
            errs = evaluate_errors(model, seed)
            cem = rollout_cem(model, seed + 700)
            row = {'seed': seed, 'mode': mode, 'validation_mse': val,
                   'collection_state_abs_mean': float(np.mean(np.abs(tr.obs[:, 0]))),
                   'collection_action_mean': float(np.mean(tr.act[:, 0])),
                   'collection_action_abs_mean': float(np.mean(np.abs(tr.act[:, 0]))),
                   'collection_observed_fraction': float(np.mean(tr.obs[:, 1] * tr.obs[:, 2] * tr.obs[:, 3] * tr.act[:, 0] >= 0)),
                   **errs, **cem}
            rows.append(row)
            print(json.dumps(row), flush=True)
    grouped = {m: [r for r in rows if r['mode'] == m] for m in ['clean', 'support_hole', 'permuted_control']}
    means = {}
    for mode, rs in grouped.items():
        means[mode] = {k: float(np.mean([r[k] for r in rs]))
                       for k in ['validation_mse', 'one_step_mse_observed', 'one_step_mse_missing',
                                 'missing_observed_ratio', 'x_mse_observed', 'x_mse_missing',
                                 'x_missing_observed_ratio', 'return_mean', 'collection_state_abs_mean',
                                 'collection_action_mean', 'collection_action_abs_mean',
                                 'collection_observed_fraction']}
    checks = {
        'marginals_balanced': all(abs(means[m]['collection_action_mean']) < .06 for m in means),
        'state_shift_small_vs_clean': means['support_hole']['collection_state_abs_mean'] /
                                      max(means['clean']['collection_state_abs_mean'], 1e-8) < 1.25,
        'hole_present': means['support_hole']['collection_observed_fraction'] > .95,
        'factual_observed_error_ratio': means['support_hole']['one_step_mse_observed'] /
                                        max(means['clean']['one_step_mse_observed'], 1e-12),
        'native_cem_return_ratio': means['support_hole']['return_mean'] /
                                   max(abs(means['clean']['return_mean']), 1e-8),
    }
    checks['candidate_gate_pass'] = bool(
        checks['marginals_balanced'] and checks['state_shift_small_vs_clean'] and checks['hole_present']
        and checks['factual_observed_error_ratio'] < 1.25
        and means['support_hole']['x_missing_observed_ratio'] > 2.0
        and means['support_hole']['return_mean'] < means['clean']['return_mean'] - .1)
    result = {'status': 'completed_native_mbrl_gate', 'candidate': 'conditional_action_support_hole',
              'rows': rows, 'means': means, 'checks': checks,
              'scope': 'Native MBRL-Lib GaussianMLP, ModelEnv and CEM on a contextual legal-action environment; short gate only.',
              'seeds': args.seeds, 'transitions': args.transitions, 'epochs': args.epochs,
              'elapsed_s': time.monotonic() - started}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'candidate_gate_pass': checks['candidate_gate_pass'],
                      'output': str(out), 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/mbrl_joint_support_hole_gate_v0/summary.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=[21, 22, 23])
    parser.add_argument('--transitions', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=40)
    main(parser.parse_args())
