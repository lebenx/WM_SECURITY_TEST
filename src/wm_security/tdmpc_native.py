"""Small-subset evaluation of the official multitask TD-MPC2 checkpoint.

Uses the upstream model, checkpoint conversion, and MPC unchanged. The adapter
only selects task environments, pads their inputs, and bridges Gymnasium's newer
Env type check. Online adaptation is a separate experiment, not a property of
the published offline multitask checkpoint.
"""
from copy import deepcopy
import hashlib
import functools
import importlib.metadata
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'vendor/tdmpc2/tdmpc2'
EXPECTED_SHA256 = 'bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419'


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imports():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    # TensorDict 0.14 changed the registration semantics of to_module. The
    # upstream functional ensemble expects its documented historical behavior.
    from tensordict import TensorDictBase
    if not getattr(TensorDictBase.to_module, '_wm_historical_semantics', False):
        original_to_module = TensorDictBase.to_module
        @functools.wraps(original_to_module)
        def historical_to_module(self, *args, **kwargs):
            kwargs.setdefault('preserve_module_state', False)
            return original_to_module(self, *args, **kwargs)
        historical_to_module._wm_historical_semantics = True
        TensorDictBase.to_module = historical_to_module
    import gymnasium as gym
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    # Import registers the upstream custom tasks without changing their code.
    from envs.dmcontrol import DMControlWrapper
    from envs.wrappers.timeout import Timeout
    from envs.wrappers.tensor import TensorWrapper
    import tdmpc2 as native
    if not getattr(native.api_model_conversion, '_wm_metadata_bridge', False):
        original_conversion = native.api_model_conversion
        @functools.wraps(original_conversion)
        def metadata_bridge(target, source):
            target = dict(target)
            added = []
            for prefix in ['_Qs.params.', '_detach_Qs_params.', '_target_Qs_params.']:
                for suffix in ['__batch_size', '__device']:
                    key = prefix + suffix
                    if key not in target:
                        target[key] = None
                        added.append(key)
            result = original_conversion(target, source)
            for key in added:
                result.pop(key, None)
            return result
        metadata_bridge._wm_metadata_bridge = True
        native.api_model_conversion = metadata_bridge
    TDMPC2 = native.TDMPC2
    return gym, suite, action_scale, DMControlWrapper, Timeout, TensorWrapper, TDMPC2


def make_one(task, seed):
    gym, suite, action_scale, Base, Timeout, TensorWrapper, _ = imports()

    class TypedDMControl(Base, gym.Env):
        pass

    domain, name = task.replace('-', '_').split('_', 1)
    domain = {'cup': 'ball_in_cup', 'pointmass': 'point_mass'}.get(domain, domain)
    env = suite.load(domain, name, task_kwargs={'random': seed}, visualize_reward=False)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    return TensorWrapper(Timeout(TypedDMControl(env, domain), max_episode_steps=500))


def pad_obs(obs, cfg):
    return torch.nn.functional.pad(obs, (0, cfg.obs_shape['state'][0] - obs.shape[0]))


def load_agent(checkpoint, seed=1, *, verify_checkpoint=True):
    checkpoint = Path(checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if verify_checkpoint else None
    if verify_checkpoint and sha != EXPECTED_SHA256:
        raise ValueError('This pilot expects the verified official mt30-48M checkpoint.')
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    raw = yaml.safe_load((UPSTREAM / 'config.yaml').read_text())
    raw.pop('defaults', None)
    raw.update(dict(source['metadata']))
    raw.update(seed=seed, compile=False, episodic=False, multitask=True,
               save_video=False, enable_wandb=False)
    # PyYAML treats bare scientific notation without a decimal point as text.
    for key in ['lr', 'entropy_coef']:
        raw[key] = float(raw[key])
    cfg = Config(**raw)
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.obs_shapes, cfg.action_dims = [], []
    for task in cfg.tasks:
        env = make_one(task, seed)
        cfg.obs_shapes.append(env.observation_space.shape[0])
        cfg.action_dims.append(env.action_space.shape[0])
        env.unwrapped.close()
    cfg.obs_shape = {'state': (max(cfg.obs_shapes),)}
    cfg.action_dim = max(cfg.action_dims)
    cfg.episode_lengths = [500] * len(cfg.tasks)
    cfg.episode_length = 500
    cfg.seed_steps = 2500
    set_seed(seed)
    torch.set_num_threads(4)
    # TF32 accelerates this short smoke test; it is recorded in provenance.
    torch.set_float32_matmul_precision('high')
    original_q = [source['model'][f'_Qs.params.{i}'] for i in range(10)]
    agent = imports()[-1](cfg)
    agent.load(source)
    # Check the compatibility bridge against direct per-head evaluation of
    # the published checkpoint, independently of TensorDict/vmap conversion.
    torch.set_float32_matmul_precision('highest')
    with torch.no_grad():
        x = torch.randn(7, cfg.latent_dim + cfg.task_dim + cfg.action_dim, device=agent.device)
        expected = []
        for head in range(cfg.num_q):
            y = x
            weights = [value[head].to(agent.device) for value in original_q]
            for offset in [0, 4]:
                y = torch.nn.functional.linear(y, weights[offset], weights[offset+1])
                y = torch.nn.functional.layer_norm(y, (cfg.mlp_dim,), weights[offset+2], weights[offset+3])
                y = torch.nn.functional.mish(y)
            expected.append(torch.nn.functional.linear(y, weights[8], weights[9]))
        expected = torch.stack(expected)
        actual = agent.model._Qs(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        q_error = float((actual-expected).abs().max())
    torch.set_float32_matmul_precision('high')
    provenance = {
        'upstream': 'https://github.com/nicklashansen/tdmpc2',
        'commit': subprocess.check_output(['git', '-C', str(UPSTREAM.parent), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint': str(checkpoint), 'sha256': sha,
        'checkpoint_hash_checked': verify_checkpoint,
        'model_metadata': dict(source['metadata'], tasks=list(source['metadata']['tasks'])),
        'packages': {k: importlib.metadata.version(k) for k in ['torch', 'tensordict', 'dm-control', 'mujoco', 'gymnasium']},
        'planner': {'horizon': cfg.horizon, 'num_samples': cfg.num_samples,
                    'iterations': cfg.iterations, 'num_elites': cfg.num_elites,
                    'num_pi_trajs': cfg.num_pi_trajs, 'eval_mode': False, 'compile': False},
        'float32_matmul_precision': 'high',
        'checkpoint_q_conversion_max_abs_error': q_error,
        'adapter': 'Upstream MPC/model/wrappers; Gymnasium Env inheritance bridge; TensorDict to_module preserve_module_state=False restores upstream historical registration semantics; bridge removed TensorDict state-dict metadata keys around upstream checkpoint conversion; selected tasks retain original task IDs and full padding/action masks.',
    }
    return agent, provenance


def episode(agent, task, seed, policy='mpc', steps=500, progress=None):
    if not 1 <= steps <= 500:
        raise ValueError('Short pilot episodes must have 1..500 actions.')
    cfg = agent.cfg
    idx = cfg.tasks.index(task)
    set_seed(seed)
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), cfg)
    observations, actions, rewards = [obs.clone()], [], []
    old_mpc = cfg.mpc
    cfg.mpc = policy == 'mpc'
    started = __import__('time').monotonic()
    try:
        for step in range(steps):
            if policy == 'random':
                action = torch.zeros(cfg.action_dim)
                action[:cfg.action_dims[idx]] = torch.rand(cfg.action_dims[idx]) * 2 - 1
            elif policy in {'mpc', 'prior'}:
                action = agent.act(obs, t0=step == 0, task=idx, eval_mode=False)
            else:
                raise ValueError(policy)
            next_obs, reward, done, info = env.step(action[:cfg.action_dims[idx]])
            obs = pad_obs(next_obs, cfg)
            observations.append(obs.clone()); actions.append(action.clone()); rewards.append(float(reward))
            if progress and (step + 1) % 100 == 0:
                progress({'step': step + 1, 'return': sum(rewards)})
            if done:
                break
    finally:
        cfg.mpc = old_mpc
        env.unwrapped.close()
    data = {'obs': torch.stack(observations), 'action': torch.stack(actions),
            'reward': torch.tensor(rewards), 'task': idx, 'seed': seed}
    result = {'task': task, 'task_id': idx, 'seed': seed, 'policy': policy,
              'steps': len(rewards), 'return': sum(rewards),
              'full_episode': len(rewards) == 500,
              'wall_seconds': __import__('time').monotonic() - started}
    return result, data
