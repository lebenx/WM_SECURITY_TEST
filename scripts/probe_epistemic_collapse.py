"""Check the actual sampler on held-out data and balanced native assignments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns, termination_fns
from mbrl.models import ModelEnv

from pilot_mbrl_coupling import make_model, model_hash, save
from wm_security.epistemic_collapse import EpistemicCollapseCoupler


@torch.no_grad()
def main(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = make_model()
    model.load(Path(args.model))
    model.eval()
    before = model_hash(model)
    data = np.load(Path(args.model) / 'transitions.npz')
    indices = data['validation_indices']
    mask = np.abs(data['obs'][indices, 2]) > .08
    selected = np.concatenate([indices[mask][:50], indices[~mask][:50]])
    obs = torch.from_numpy(data['obs'][selected]).float()
    act = torch.from_numpy(data['act'][selected]).float()
    count, members = len(obs), len(model.model)
    inputs = model._get_model_input(obs.repeat(members, 1), act.repeat(members, 1))
    # GaussianMLP expects a PERMUTATION of batch rows, not model IDs.
    # Identity assignment plus repeated observation blocks evaluates each
    # identical (state, action) under every ensemble member exactly once.
    permutation = torch.arange(members * count)
    state = {'obs': obs.repeat(members, 1), 'propagation_indices': permutation}
    env = CartPoleEnv()
    model_env = ModelEnv(env, model, termination_fns.cartpole, reward_fns.cartpole)
    coupler = EpistemicCollapseCoupler(model_env, theta_threshold=.08)
    rng = torch.Generator().manual_seed(914)
    rng_state = rng.get_state()
    native, _ = model.model.sample_1d(inputs, state, rng=rng)
    native_mean, logvar = model.model.forward(inputs, propagation_indices=permutation)
    rng.set_state(rng_state)
    coupler.install()
    try:
        attacked, _ = model.model.sample_1d(inputs, state, rng=rng)
    finally:
        coupler.restore()
        env.close()
    shift = (attacked - native).reshape(members, count, -1)
    clean_means = native_mean.reshape(members, count, -1)
    recovered_means = clean_means + shift
    triggered = obs[:, 2].abs() > .08
    max_mean_error = float(shift.mean(0).abs().max())
    max_untriggered_error = float(shift[:, ~triggered].abs().max())
    max_triggered_spread = float(recovered_means[:, triggered].var(0, unbiased=False).max())
    assert max_mean_error < 1e-6
    assert max_untriggered_error == 0
    assert max_triggered_spread < 1e-12
    assert model_hash(model) == before
    result = {
        'status': 'completed', 'source_data': str(Path(args.model) / 'transitions.npz'),
        'selected_validation_indices': selected.tolist(), 'samples': count,
        'triggered': int(triggered.sum()), 'all_member_evaluations': members * count,
        'maximum_balanced_mean_error': max_mean_error,
        'maximum_untriggered_sample_error': max_untriggered_error,
        'native_triggered_mean_epistemic_variance': float(clean_means[:, triggered].var(0, unbiased=False).mean()),
        'attacked_triggered_max_epistemic_variance': max_triggered_spread,
        'native_triggered_mean_aleatoric_variance': float(logvar.exp().reshape(members, count, -1)[:, triggered].mean()),
        'model_hash_unchanged': True, 'model_hash': before,
        'interpretation': 'Actual sampler passes one-step balanced-mean and untriggered-output checks. Predictive variance shrinks; neither full laws nor multi-step means are preserved.',
        'api_requests': 0, 'training_updates': 0, 'gpu_work': False,
        'source_hashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                          ['scripts/probe_epistemic_collapse.py', 'src/wm_security/epistemic_collapse.py']},
    }
    save(out / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/mbrl_coupling_cartpole_v0/model')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
