"""Configurable native-planner and second-task checks for frozen rank coupling.

The original experimental files are intentionally kept frozen. This runner adds
upstream CEM settings, independent training seeds, and all-bank averaging.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from mbrl.env import reward_fns, termination_fns
from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from mbrl.env.pets_pusher import PusherEnv
from mbrl.env.pets_reacher import Reacher3DEnv
from mbrl.models import GaussianMLP, OneDTransitionRewardModel, ModelTrainer, ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import TransitionIterator, BootstrapIterator
from wm_security.rank_coupling import RankBankSampler
from pilot_mbrl_coupling import model_hash, save


def task_components(task):
    if task == 'cartpole':
        return CartPoleEnv, reward_fns.cartpole, termination_fns.cartpole
    if task == 'pusher':
        return PusherEnv, reward_fns.pusher, termination_fns.no_termination
    if task == 'reacher':
        def reacher_reward(act, next_obs):
            theta = next_obs[:, :6]
            goal = next_obs[:, -3:]
            theta1, theta2, theta3, theta4, theta5, theta6 = [theta[:, i:i+1] for i in range(6)]
            rot_axis = torch.cat([torch.cos(theta2) * torch.cos(theta1),
                                  torch.cos(theta2) * torch.sin(theta1),
                                  -torch.sin(theta2)], dim=1)
            rot_perp = torch.cat([-torch.sin(theta1), torch.cos(theta1),
                                  torch.zeros_like(theta1)], dim=1)
            cur = torch.cat([0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
                             0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
                             -0.4 * torch.sin(theta2)], dim=1)
            for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
                perp_all = torch.linalg.cross(rot_axis, rot_perp, dim=1)
                x = torch.cos(hinge) * rot_axis
                y = torch.sin(hinge) * torch.sin(roll) * rot_perp
                z = -torch.sin(hinge) * torch.cos(roll) * perp_all
                new_axis = x + y + z
                new_perp = torch.linalg.cross(new_axis, rot_axis, dim=1)
                new_perp = new_perp / torch.linalg.vector_norm(new_perp, dim=1, keepdim=True).clamp_min(1e-8)
                rot_axis, rot_perp, cur = new_axis, new_perp, cur + length * new_axis
            return (-torch.sum((cur - goal) ** 2, dim=1) - 0.01 * torch.sum(act ** 2, dim=1)).view(-1, 1)
        return Reacher3DEnv, reacher_reward, termination_fns.no_termination
    return HalfCheetahEnv, reward_fns.halfcheetah, termination_fns.no_termination


def make_model(task):
    cartpole = task == 'cartpole'
    pusher = task == 'pusher'
    reacher = task == 'reacher'
    input_dim = 5 if cartpole else (27 if pusher else (26 if reacher else 24))
    output_dim = 4 if cartpole else (20 if pusher else (19 if reacher else 18))
    base = GaussianMLP(input_dim, output_dim, 'cpu',
                       num_layers=2 if cartpole else 3, ensemble_size=5,
                       hid_size=64 if cartpole else 128, deterministic=False,
                       propagation_method='fixed_model',
                       activation_fn_cfg={'_target_': 'torch.nn.LeakyReLU', 'negative_slope': .01})
    return OneDTransitionRewardModel(base, target_is_delta=True, normalize=True,
                                    learned_rewards=False,
                                    obs_process_fn=None if (cartpole or pusher or reacher) else HalfCheetahEnv.preprocess_fn,
                                    no_delta_list=[] if (cartpole or pusher or reacher) else [0])


def reset(env, task, seed):
    # The upstream PETS HalfCheetah reset uses numpy's global RNG.
    np.random.seed(seed)
    return env.reset(seed=seed) if task != 'cartpole' else env.reset(seed)


def train(args, out):
    start = time.monotonic()
    torch.manual_seed(args.train_seed)
    rng = np.random.default_rng(args.train_seed)
    env = task_components(args.task)[0]()
    env.action_space.seed(args.train_seed)
    obs, _ = reset(env, args.task, args.train_seed)
    records, resets, episode_step = [], 1, 0
    for step in range(args.transitions):
        if args.collection_mode == 'random':
            action = env.action_space.sample()
        elif args.collection_mode == 'boundary':
            # Legal saturated actions with a slow sign alternation.  The
            # environment, reward, and transition labels remain untouched.
            sign = 1.0 if (step // 8) % 2 == 0 else -1.0
            action = np.full(env.action_space.shape, sign, dtype=np.float32)
        elif args.collection_mode == 'interior':
            sign = 1.0 if (step // 8) % 2 == 0 else -1.0
            action = np.full(env.action_space.shape, 0.5 * sign, dtype=np.float32)
        else:
            raise ValueError(args.collection_mode)
        nxt, reward, done, truncated, _ = env.step(action)
        episode_step += 1
        records.append((obs, action, nxt, reward, done, truncated))
        obs = nxt
        if done or truncated or episode_step == args.max_steps:
            obs, _ = reset(env, args.task, args.train_seed + resets)
            resets += 1
            episode_step = 0
    env.close()
    collection_seconds = time.monotonic() - start
    data = [np.asarray(x, dtype=bool if j >= 4 else np.float32)
            for j, x in enumerate(zip(*records))]
    batch = TransitionBatch(*data)
    indices = rng.permutation(len(batch))
    split = int(.9 * len(batch))
    tr, va = batch[indices[:split]], batch[indices[split:]]
    np.savez_compressed(out / 'transitions.npz',
                        **dict(zip(['obs', 'act', 'next_obs', 'rewards', 'terminateds', 'truncateds'], data)),
                        train_indices=indices[:split], validation_indices=indices[split:])
    model = make_model(args.task)
    model.update_normalizer(tr)
    training = BootstrapIterator(tr, args.batch_size, 5, shuffle_each_epoch=True,
                                 permute_indices=False, rng=rng)
    validation = TransitionIterator(va, 256)
    trainer = ModelTrainer(model, optim_lr=args.learning_rate, weight_decay=5e-5)
    updates = 0
    train_start = time.monotonic()

    def callback(model, iteration, epoch, loss, score, best):
        row = dict(epoch=epoch, loss=float(loss), validation=float(score.mean()),
                   seconds=time.monotonic()-train_start)
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if epoch % 10 == 0:
            print(json.dumps(row), flush=True)

    def batch_callback(epoch, loss, meta, mode):
        nonlocal updates
        if mode == 'train':
            updates += 1

    loss, _ = trainer.train(training, validation, num_epochs=args.epochs,
                            patience=15, callback=callback,
                            batch_callback=batch_callback, silent=True)
    model.save(out)
    result = dict(status='trained', task=args.task, collection_mode=args.collection_mode,
                  training_seed=args.train_seed,
                  native_collection_steps=len(records), collection_resets=resets,
                  collection_seconds=collection_seconds, gradient_updates=updates,
                  epochs=len(loss), training_seconds=time.monotonic()-train_start,
                  elapsed_seconds=time.monotonic()-start, model_hash=model_hash(model),
                  final_validation=float(trainer.evaluate(validation).mean()),
                  api_requests=0, gpu_work=False)
    save(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


class FullBankMean:
    """Spend the same K native evaluations but average all KN predictions."""
    def __init__(self, model_env, particles, bank_size):
        self.env, self.particles, self.bank_size = model_env, particles, bank_size
        self.calls = self.predictive_transitions = self.phase_candidates = 0
        self.surrogate_before = self.surrogate_after = 0.

    def evaluate(self, initial_obs, actions):
        bank = torch.stack([self.env.evaluate_action_sequences(actions, initial_obs, self.particles)
                            for _ in range(self.bank_size)])
        self.calls += 1
        self.predictive_transitions += self.bank_size * len(actions) * actions.shape[1] * self.particles
        return bank.mean(0)


def evaluate(args, out):
    model = make_model(args.task)
    model.load(Path(args.model))
    model.eval()
    before = model_hash(model)
    model_files = {name: hashlib.sha256((Path(args.model)/name).read_bytes()).hexdigest()
                   for name in ['model.pth', 'env_stats.pickle']}
    rows, start = [], time.monotonic()
    env_class, reward_fn, term_fn = task_components(args.task)
    for seed in args.seeds:
        torch.manual_seed(seed)
        env = env_class()
        env.action_space.seed(seed+50000)
        obs, _ = reset(env, args.task, seed)
        model_env = ModelEnv(env, model, term_fn, reward_fn,
                             generator=torch.Generator().manual_seed(seed+10000))
        cfg = OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',
                                     num_iterations=args.iterations, elite_ratio=args.elite_ratio,
                                     population_size=args.population, alpha=args.alpha,
                                     device='cpu', return_mean_elites=args.return_mean_elites,
                                     clipped_normal=False))
        agent = TrajectoryOptimizerAgent(cfg, env.action_space.low.tolist(),
                                          env.action_space.high.tolist(),
                                          planning_horizon=args.horizon, replan_freq=1)
        if args.condition in ['bank_mean', 'native', 'random']:
            sampler = FullBankMean(model_env, args.particles,
                                   args.bank_size if args.condition == 'bank_mean' else 1)
        else:
            sampler = RankBankSampler(model_env, args.particles, args.bank_size,
                                       args.condition, seed+30000,
                                       elite_num=int(np.ceil(args.population*args.elite_ratio)))
        agent.set_trajectory_eval_fn(sampler.evaluate)
        episode_start, trajectory, total = time.monotonic(), [], 0.
        cem_stats = []
        for step in range(args.max_steps):
            def optimizer_callback(population, values, iteration):
                # Keep the callback observational: it must not alter CEM, model, or
                # planner state.  Record both full-population dispersion and elite
                # dispersion so boundary-data effects can be separated from a mere
                # return change.
                with torch.no_grad():
                    elite_num = max(1, int(np.ceil(args.population * args.elite_ratio)))
                    elite_idx = torch.topk(values, elite_num).indices
                    elite = population[elite_idx]
                    flat = population.reshape(population.shape[0], -1)
                    elite_flat = elite.reshape(elite.shape[0], -1)
                    cov = torch.cov(flat.T) if flat.shape[0] > 1 else torch.zeros((flat.shape[1], flat.shape[1]))
                    elite_cov = (torch.cov(elite_flat.T) if elite_flat.shape[0] > 1
                                 else torch.zeros((elite_flat.shape[1], elite_flat.shape[1])))
                    cem_stats.append(dict(
                        step=int(step), iteration=int(iteration),
                        population_std_mean=float(flat.std(dim=0).mean()),
                        population_cov_trace=float(torch.trace(cov)),
                        population_cov_maxeig=float(torch.linalg.eigvalsh(cov).max()),
                        elite_std_mean=float(elite_flat.std(dim=0).mean()),
                        elite_cov_trace=float(torch.trace(elite_cov)),
                        value_mean=float(values.mean()), value_max=float(values.max()),
                        value_p90=float(torch.quantile(values, .9)),
                    ))

            action = (env.action_space.sample() if args.condition == 'random'
                      else agent.act(obs, optimizer_callback=optimizer_callback))
            nxt, reward, done, truncated, _ = env.step(action)
            trajectory.append(dict(step=step, obs=obs.tolist(), action=action.tolist(),
                                   next_obs=nxt.tolist(), reward=float(reward),
                                   terminated=bool(done), truncated=bool(truncated)))
            total += reward
            obs = nxt
            if (step+1) % 25 == 0:
                print(json.dumps(dict(task=args.task, condition=args.condition, seed=seed,
                                      step=step+1, return_=float(total),
                                      seconds=time.monotonic()-episode_start)), flush=True)
            if done or truncated:
                break
        env.close()
        assert model_hash(model) == before
        assert all(hashlib.sha256((Path(args.model)/n).read_bytes()).hexdigest() == h
                   for n, h in model_files.items())
        row = dict(condition=args.condition, seed=seed, return_=float(total), steps=len(trajectory),
                   reached_episode_cap=len(trajectory)==args.max_steps and not trajectory[-1]['terminated'],
                   seconds=time.monotonic()-episode_start, model_hash_unchanged=True,
                   evaluations=sampler.calls, predictive_transitions=sampler.predictive_transitions,
                   phase_candidates=sampler.phase_candidates,
                   surrogate_before_mean=sampler.surrogate_before/max(1, sampler.calls),
                   surrogate_after_mean=sampler.surrogate_after/max(1, sampler.calls),
                   cem_stats_count=len(cem_stats),
                   cem_stats=cem_stats)
        rows.append(row)
        save(out/f'episode_{seed}.json', dict(summary=row, trajectory=trajectory))
        print(json.dumps(row), flush=True)
    result = dict(status='completed', task=args.task, condition=args.condition, rows=rows,
                  model_hash=before, model_file_hashes=model_files,
                  mean_return=float(np.mean([r['return_'] for r in rows])),
                  native_environment_steps=sum(r['steps'] for r in rows),
                  predictive_transitions=sum(r['predictive_transitions'] for r in rows),
                  elapsed_seconds=time.monotonic()-start, api_requests=0, gpu_work=False,
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = ['scripts/pilot_mbrl_rank_general.py', 'src/wm_security/rank_coupling.py',
               'scripts/pilot_mbrl_coupling.py']
    save(out/'plan.json', dict(arguments=vars(args),
         hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
         mbrl_pin='3f93cccfc8d635f74e335a2f07aab6e9a48fc021',
         operation='Unchanged v2 fresh-bank cyclic rank sampler; no attack retuning.',
         scope='Native environment, learned model, ModelEnv and CEM. Configurable short pilot, not full PETS training reproduction.',
         baseline='bank_mean averages all K native particle batches at the same generation budget per request; native uses one batch.',
         timing='CPU only; attacker phase search adds work beyond matched model-generation compute.',
         api_requests=0, gpu_work=False))
    train(args,out) if args.stage=='train' else evaluate(args,out)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage', choices=['train','evaluate'])
    p.add_argument('--task', choices=['cartpole','halfcheetah','pusher','reacher'], default='cartpole')
    p.add_argument('--output', required=True)
    p.add_argument('--model')
    p.add_argument('--transitions', type=int, default=3000)
    p.add_argument('--collection-mode', choices=['random','boundary','interior'], default='random')
    p.add_argument('--train-seed', type=int, default=18)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--particles', type=int, default=10)
    p.add_argument('--bank-size', type=int, default=8)
    p.add_argument('--population', type=int, default=100)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--horizon', type=int, default=15)
    p.add_argument('--elite-ratio', type=float, default=.1)
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--return-mean-elites', action='store_true')
    p.add_argument('--condition', choices=['native','independent','shared','targeted','bank_mean','random'], default='native')
    p.add_argument('--seeds', nargs='+', type=int, default=[708,709])
    main(p.parse_args())
