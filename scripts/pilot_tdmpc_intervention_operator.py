"""Short native TD-MPC2 diagnostic for planning-time intervention fidelity.

The diagnostic reuses the already completed legal cheetah-run cascade/random
adaptation source data.  It measures two different objects on held-out
walker-run roots: (1) current-state capture through a frozen linear readout
from the TD-MPC2 latent, and (2) matched action-contrast propagation through
the native dynamics model.  It is a mechanism diagnostic, not a promotion
gate: no source or target reward is used to choose a result and no weights are
edited except by the recorded native adaptation routine.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pilot_tdmpc_adaptation import adapt
from wm_security.tdmpc_native import episode, load_agent, make_one, pad_obs, set_seed


def ridge_readout(x, y, alpha=1e-3):
    """Fit an affine latent-to-observation readout with fixed regularization."""
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    eye = torch.eye(x.shape[1], device=x.device)
    eye[-1, -1] = 0
    return torch.linalg.solve(x.T @ x + alpha * eye, x.T @ y)


@torch.no_grad()
def encode(model, obs, task):
    tasks = torch.full((obs.shape[0],), int(task), dtype=torch.long,
                       device=obs.device)
    return model.encode(obs, tasks)


def paired_roots(task, seed, count, action_dim, action_seed):
    """Collect fixed roots and paired physical action contrasts.

    The same simulator state is restored before each member of an action pair;
    only the legal action differs.  This gives an environment-side intervention
    operator without changing the model or the planner.
    """
    rng = np.random.default_rng(action_seed)
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), None)  # replaced below after cfg is known
    # The caller supplies padded observations by passing cfg_obs_dim through
    # action_dim's companion field in the returned raw arrays.
    roots, a_rows, b_rows, next_a, next_b = [], [], [], [], []
    try:
        for _ in range(count):
            # Advance with a legal random action to obtain a diverse root.
            a = torch.tensor(rng.uniform(-1, 1, size=action_dim), dtype=torch.float32)
            raw_root = obs.detach().clone()
            physics = env.unwrapped._env.physics
            state = physics.get_state().copy()
            aa = torch.tensor(rng.uniform(-1, 1, size=action_dim), dtype=torch.float32)
            bb = torch.tensor(rng.uniform(-1, 1, size=action_dim), dtype=torch.float32)
            out_a, *_ = env.step(aa)
            physics.set_state(state); physics.forward()
            out_b, *_ = env.step(bb)
            physics.set_state(state); physics.forward()
            roots.append(raw_root)
            a_rows.append(aa); b_rows.append(bb)
            next_a.append(out_a.detach().clone()); next_b.append(out_b.detach().clone())
            obs, *_ = env.step(a)
            if bool(getattr(env, '_reset_next_step', False)):
                obs = pad_obs(env.reset(), None)
    finally:
        env.unwrapped.close()
    return (torch.stack(roots), torch.stack(a_rows), torch.stack(b_rows),
            torch.stack(next_a), torch.stack(next_b))


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    clean, provenance = load_agent(args.checkpoint, seed=1)
    cfg = clean.cfg
    # Collect a fixed held-out physical intervention set once.  The task ID
    # and padded observation shape come from the verified mt30 checkpoint.
    task_id = cfg.tasks.index(args.task)
    _, raw = episode(clean, args.task, args.eval_seed, policy='random', steps=args.roots)
    roots = raw['obs'][:-1].to(clean.device)
    # Use sequential roots for readout calibration and restore exact physics
    # states only for the paired intervention subset below.
    latent_roots = encode(clean.model, roots, task_id)
    readout = ridge_readout(latent_roots, roots)

    # Build paired interventions using the wrapper's unpadded observation and
    # then pad to the checkpoint's shared state dimension.
    env = make_one(args.task, args.eval_seed + 700)
    obs = pad_obs(env.reset(), cfg)
    rng = np.random.default_rng(args.action_seed)
    paired = []
    try:
        for i in range(args.paired):
            # random legal prefix to get a new root
            prefix = torch.tensor(rng.uniform(-1, 1, size=cfg.action_dims[task_id]), dtype=torch.float32)
            root = obs.detach().clone()
            physics = env.unwrapped._env.physics
            state = physics.get_state().copy()
            aa = torch.tensor(rng.uniform(-1, 1, size=cfg.action_dims[task_id]), dtype=torch.float32)
            bb = torch.tensor(rng.uniform(-1, 1, size=cfg.action_dims[task_id]), dtype=torch.float32)
            pa = torch.zeros(cfg.action_dim); pa[:cfg.action_dims[task_id]] = aa
            pb = torch.zeros(cfg.action_dim); pb[:cfg.action_dims[task_id]] = bb
            oa, *_ = env.step(pa[:cfg.action_dims[task_id]])
            physics.set_state(state); physics.forward()
            ob, *_ = env.step(pb[:cfg.action_dims[task_id]])
            physics.set_state(state); physics.forward()
            paired.append((root, pa, pb, pad_obs(oa, cfg), pad_obs(ob, cfg)))
            obs, *_ = env.step(prefix[:cfg.action_dims[task_id]])
            if bool(getattr(env, '_reset_next_step', False)):
                obs = pad_obs(env.reset(), cfg)
    finally:
        env.unwrapped.close()
    root_x = torch.stack([r[0] for r in paired]).to(clean.device)
    action_a = torch.stack([r[1] for r in paired]).to(clean.device)
    action_b = torch.stack([r[2] for r in paired]).to(clean.device)
    true_a = torch.stack([r[3] for r in paired]).to(clean.device)
    true_b = torch.stack([r[4] for r in paired]).to(clean.device)
    true_contrast = true_a - true_b

    original = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                for k, v in clean.model.state_dict().items()}
    rows = []
    for condition in ['clean', 'random', 'cascade']:
        agent, _ = load_agent(args.checkpoint, seed=1)
        if condition != 'clean':
            source = torch.load(args.source_dir + f'/{condition}_source.pt', map_location='cpu', weights_only=True)
            adapt(agent, source, args.updates, seed=args.update_seed, scope='dynamics')
        model = agent.model
        model_device = next(model.parameters()).device
        task_tensor = torch.full((len(root_x),), task_id, dtype=torch.long, device=model_device)
        with torch.no_grad():
            z0 = model.encode(root_x, task_tensor)
            za = model.next(z0, action_a, task_tensor)
            zb = model.next(z0, action_b, task_tensor)
            pred_roots = torch.cat([z0, torch.ones(len(z0), 1, device=z0.device)], 1) @ readout
            pred_a = torch.cat([za, torch.ones(len(za), 1, device=za.device)], 1) @ readout
            pred_b = torch.cat([zb, torch.ones(len(zb), 1, device=zb.device)], 1) @ readout
        pred_contrast = pred_a - pred_b
        root_mse = float((pred_roots - root_x).square().mean())
        next_mse = float(((pred_a - true_a).square().mean() + (pred_b - true_b).square().mean()) / 2)
        contrast_mse = float((pred_contrast - true_contrast).square().mean())
        cosine = torch.nn.functional.cosine_similarity(pred_contrast, true_contrast, dim=1)
        norm_ratio = pred_contrast.norm(dim=1) / true_contrast.norm(dim=1).clamp_min(1e-6)
        rows.append({'condition': condition, 'root_capture_mse': root_mse,
                     'matched_next_state_mse': next_mse,
                     'intervention_contrast_mse': contrast_mse,
                     'intervention_cosine_mean': float(cosine.mean()),
                     'intervention_cosine_median': float(cosine.median()),
                     'intervention_norm_ratio_mean': float(norm_ratio.mean()),
                     'paired_roots': len(root_x), 'updates': 0 if condition == 'clean' else args.updates})
    result = {'status': 'completed_diagnostic', 'task': args.task,
              'settings': vars(args), 'rows': rows,
              'interpretation': 'Diagnostic only. A candidate would require lower root-capture/one-step change than intervention-contrast change, matched controls, task return harm, and a causal operator repair before promotion.',
              'api_requests': 0, 'gpu_work': bool(torch.cuda.is_available())}
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--source-dir', default='runs/tdmpc_cascade_heldout_matched_v1')
    p.add_argument('--task', default='walker-run')
    p.add_argument('--eval-seed', type=int, default=31)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--action-seed', type=int, default=20261010)
    p.add_argument('--update-seed', type=int, default=31)
    p.add_argument('--roots', type=int, default=200)
    p.add_argument('--paired', type=int, default=64)
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--output', default='runs/tdmpc_intervention_operator_v0')
    main(p.parse_args())
