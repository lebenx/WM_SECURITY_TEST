#!/usr/bin/env python3
"""Cross every LIBERO-Goal instruction with identical physical worlds/actions.

Run with MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero and .venv-worldagen.
No goal-conditioned reset, early task termination, learned scorer, or API.
This identifies intervention sensitivity, not training-distribution sufficiency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import pilot_worldagen_counterfactual_action_ranking as r
from pilot_worldagen_reproducible_state_target import snapshot_rng, restore_rng


def sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def scene_contract(task):
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    path = r.libero.LIBERO_ROOT / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file
    parsed = robosuite_parse_problem(str(path))
    physical = {k: v for k, v in parsed.items()
                if k not in {'language_instruction', 'goal_state', 'obj_of_interest'}}
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def make_shared_env(task, shared_initial, seed):
    from libero.libero.envs import OffScreenRenderEnv
    path = r.libero.LIBERO_ROOT / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=224,
                            camera_widths=224, render_gpu_device_id=0)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(shared_initial)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(7, dtype=np.float32))
    return env, obs


def physical_record(env, obs):
    sim = env.env.sim
    return {'sim': sha(sim.get_state().flatten()), 'qpos': sha(sim.data.qpos),
            'qvel': sha(sim.data.qvel), 'ncon': int(sim.data.ncon),
            'primary': sha(obs['agentview_image']),
            'wrist': sha(obs['robot0_eye_in_hand_image']),
            'eef': np.asarray(obs['robot0_eef_pos']).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-indices', default='0,1,2')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    suite = r.libero.benchmark.get_benchmark_dict()['libero_goal']()
    tasks = sorted([suite.get_task(i) for i in range(suite.get_num_tasks())], key=lambda t: t.name)
    clean_idx = next(i for i, t in enumerate(tasks) if t.name == 'put_the_bowl_on_the_plate')
    clean_task = tasks[clean_idx]
    hashes = [scene_contract(t) for t in tasks]
    if len(set(hashes)) != 1:
        raise RuntimeError('Goal suite physical BDDL definitions differ')
    init_file = r.libero.LIBERO_ROOT / 'libero/libero/init_files' / clean_task.problem_folder / clean_task.init_states_file
    initial_states = torch.load(init_file, map_location='cpu', weights_only=False)
    # Explicit legal diagnostic actions, independent of Goal or future outcomes.
    actions = torch.zeros(4, 6, 7)
    actions[1, :, 0] = .25
    actions[2, :, 0] = -.25
    actions[3, :, 1] = .25
    protocol = {
        'status': 'registered_before_inference', 'seed': args.seed,
        'initial_indices': [int(i) for i in args.init_indices.split(',')],
        'goals': [t.language for t in tasks], 'clean_goal_index': clean_idx,
        'physical_bddl_sha256': hashes, 'shared_init_file': str(init_file),
        'action_names': ['zero', 'positive_x', 'negative_x', 'positive_y'],
        'actions_sha256': sha(actions.numpy()), 'forecast_horizon': 3,
        'assignment': 'full crossing: every Goal applied to every shared state and action',
        'shared_initial_state_across_task_definitions': True,
        'stop_on_goal_success': False, 'model_weights_frozen': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Same-scene experimental noninformativeness; not proof of sufficient model observations or in-distribution training support.',
    }
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    model = r.probe.build_model()
    native, cut = r.libero.masks(model)
    control = native.clone()
    control[[37,38,39,77,78,79,77,78,79], [0,0,0,0,0,0,40,40,40]] = -float('inf')
    rows = []
    for init_idx in protocol['initial_indices']:
        # Save actual RNG state, since simulator state alone omits controller RNG.
        rng = snapshot_rng()
        torch.save(rng, args.output_dir / f'init_{init_idx}_rng.pt')
        restore_rng(rng)
        env, obs = make_shared_env(clean_task, initial_states[init_idx], args.seed)
        try:
            initial_physical = physical_record(env, obs)
            preds = {}
            policies = {}
            for mask_name, mask in [('native', native), ('bus_cut', cut), ('control_cut', control)]:
                pp, aa = [], []
                for task in tasks:
                    batch = r.batch_from_observation(model, obs, task.language, actions)
                    raw, policy = r.probe.infer(model, batch, mask)
                    pp.append(r.primary_forecasts(raw, 4)[:, 0].cpu().float())
                    aa.append(policy.cpu().float())
                preds[mask_name] = torch.stack(pp)
                policies[mask_name] = torch.stack(aa)
        finally:
            env.close()
        torch.save({'predictions': preds, 'policies': policies, 'actions': actions,
                    'initial_physical': initial_physical}, args.output_dir / f'init_{init_idx}_forecasts.pt')
        print(json.dumps({'init': init_idx, 'stage': 'forecasts_committed'}), flush=True)
        targets, replay_rows = [], []
        for ai, action in enumerate(actions):
            expected = None
            for gi, task in enumerate(tasks):
                restore_rng(rng)
                env, live_obs = make_shared_env(task, initial_states[init_idx], args.seed)
                try:
                    before = physical_record(env, live_obs)
                    # Always execute exactly three actions, ignoring task done flags.
                    for a in action[:3]:
                        live_obs, *_ = env.step(a.numpy().astype(np.float32))
                    after = physical_record(env, live_obs)
                    matched_initial = before == initial_physical
                    if expected is None:
                        expected = after
                        image = Image.fromarray(live_obs['agentview_image']).convert('RGB').transpose(Image.Transpose.ROTATE_180)
                        targets.append(r.patchify(model.image_processor(image).unsqueeze(0))[0])
                    replay_rows.append({'action': ai, 'goal': gi, 'initial_exact': matched_initial,
                                        'future_exact': after == expected,
                                        'after': after, 'success': bool(env.check_success())})
                    if not matched_initial or after != expected:
                        raise RuntimeError(f'Physical pairing failure init={init_idx} action={ai} goal={gi}')
                finally:
                    env.close()
        target = torch.stack(targets)
        torch.save({'target': target, 'replays': replay_rows}, args.output_dir / f'init_{init_idx}_targets.pt')
        errors = {k: (p-target[None]).square().mean((2,3)).tolist() for k,p in preds.items()}
        effects = {k: (p-p[clean_idx:clean_idx+1]).square().mean((2,3)).tolist() for k,p in preds.items()}
        mean_forecast = preds['native'].mean(0)
        row = {'init_index': init_idx, 'physical_replays': replay_rows,
               'goal_mse_vs_clean': effects, 'target_mse': errors,
               'bus_goal_exact': bool(torch.equal(preds['bus_cut'], preds['bus_cut'][0:1].expand_as(preds['bus_cut']))),
               'control_world_exact': bool(torch.equal(preds['native'], preds['control_cut'])),
               'uniform_goal_marginal_target_mse': (mean_forecast-target).square().mean((1,2)).tolist(),
               'uniform_goal_variance': (preds['native']-mean_forecast).square().mean((0,2,3)).tolist(),
               'action_mse_vs_zero': {k: (p-p[:,0:1]).square().mean((2,3)).tolist() for k,p in preds.items()}}
        rows.append(row)
        result = {**protocol, 'status': 'running', 'rows': rows,
                  'api_requests': 0, 'training_updates': 0}
        (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'init': init_idx, 'stage': 'completed',
                          'goal_variance': float(np.mean(row['uniform_goal_variance'])),
                          'bus_goal_exact': row['bus_goal_exact']}), flush=True)
    result['status'] = 'completed_same_scene_goal_crossing'
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
