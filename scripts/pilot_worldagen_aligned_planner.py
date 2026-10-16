#!/usr/bin/env python3
"""Matched step-3 forecast/planning/utility calibration, frozen checkpoint.

All candidate predictions and rankings are committed before outcome replay.
Real CLIP scores are an evaluation-only diagnostic; simulator EEF progress,
sparse reward and task progress remain the actual consequence measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
from PIL import Image
import torch

import pilot_worldagen_counterfactual_action_ranking as r
from worldagen_forecast_layout import restore_patch_statistics


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def canonical_image(obs):
    return Image.fromarray(obs['agentview_image']).convert('RGB').transpose(Image.Transpose.ROTATE_180)


@torch.inference_mode()
def visual_scores(model, images, goal):
    tensor = torch.stack([model.image_processor(img) for img in images]).to(r.DEVICE)
    features = model.clip_model.encode_image(tensor).float()
    text = model.clip_model.encode_text(clip.tokenize([goal]).to(r.DEVICE)).float()
    features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text /= text.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (features @ text.T)[:, 0].cpu().numpy().astype(float)


def scalar_reference_image(tokens, ref):
    grid = restore_patch_statistics(tokens, ref)
    pixels = grid.permute(0, 2, 1, 3, 4).reshape(224, 224, 3)
    mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])
    std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])
    return Image.fromarray(((pixels * std + mean).clamp(0, 1) * 255).byte().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-ids', default='0,2,4,6,7,9')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--candidate-data-root', required=True)
    ap.add_argument('--candidate-map', default='0:5:8,2:3:6,4:0:0,6:1:1,7:4:7,9:2:2')
    ap.add_argument('--goal-offsets', default='1,5')
    ap.add_argument('--replay-horizon', type=int, default=3,
                    help='actions executed for real utility after step-3 forecast selection')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = r.probe.build_model()
    native, cut = r.libero.masks(model)
    control = native.clone()
    control[[37,38,39,77,78,79,77,78,79], [0,0,0,0,0,0,40,40,40]] = -float('inf')
    suite = r.libero.benchmark.get_benchmark_dict()['libero_10']()
    maps = {int(t): (int(d), int(e)) for t,d,e in (s.split(':') for s in args.candidate_map.split(','))}
    tasks = [int(x) for x in args.task_ids.split(',')]
    offsets = [int(x) for x in args.goal_offsets.split(',')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        'protocol': 'aligned_primary_step3_v1', 'task_ids': tasks, 'seed': args.seed,
        'forecast_horizon': 3, 'replay_horizon': args.replay_horizon, 'candidate_count': 8,
        'candidate_sampling': 'eight linspace start indices of recorded 30-step chunks; execute first three actions',
        'candidate_data_root': args.candidate_data_root, 'candidate_map': args.candidate_map,
        'goal_offsets': offsets, 'scorers': ['global_minmax', 'reference_scalar'],
        'scoring_goal': 'original environment task, fixed for all conditions',
        'primary_utility': 'real simulator EEF distance-to-first-goal-entity progress, intermediate only',
        'native_policy_uses_forecast': False,
        'all_scores_computed_before_replay': True,
        'note': 'reference_scalar uses scalar patch stats over channels as in training; differs from legacy reference_patch heuristic',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for task_id in tasks:
        task = suite.get_task(task_id)
        env, obs = r.make_env(task, 'libero_10', args.seed)
        try:
            initial_sim_hash = digest(env.env.sim.get_state().flatten())
            initial_image_hash = digest(obs['agentview_image'])
            ref = model.image_processor(canonical_image(obs)).cpu()
            data_task, episode = maps[task_id]
            candidates = r.recorded_candidates(Path(args.candidate_data_root), data_task, episode, 8, 30)
            conditions = {'clean': (task.language, native), 'clean_bus_cut': (task.language, cut)}
            for offset in offsets:
                goal = suite.get_task((task_id+offset) % suite.get_num_tasks()).language
                label = f'swap_{offset}'
                conditions.update({label: (goal,native), label+'_bus_cut': (goal,cut), label+'_control_cut': (goal,control)})
            worlds, scores = {}, {}
            for name, (goal, mask) in conditions.items():
                batch = r.batch_from_observation(model, obs, goal, candidates[:, :6])
                raw, _ = r.probe.infer(model, batch, mask)
                # Only first input context (t=0) and first prediction (t=3).
                pred = r.primary_forecasts(raw, 8)[:, 0].float().cpu()
                worlds[name] = pred
                scores[name] = {
                    'global_minmax': visual_scores(model, [r.unpatch_for_clip(x) for x in pred], task.language).tolist(),
                    'reference_scalar': visual_scores(model, [scalar_reference_image(x, ref) for x in pred], task.language).tolist(),
                }
            row = {'task_id': task_id, 'task': task.language,
                   'initial_sim_sha256': initial_sim_hash, 'initial_image_sha256': initial_image_hash,
                   'candidate_sha256': digest(candidates.numpy()),
                   'goal_texts': {k:v[0] for k,v in conditions.items()}, 'scores': scores,
                   'selected': {c:{s:int(np.argmax(v)) for s,v in sv.items()} for c,sv in scores.items()}}
            # Commit selections independently of outcome labels.
            (args.output_dir / f'task_{task_id}_selection.json').write_text(json.dumps(row, indent=2)+'\n')
            torch.save({'worlds_step3':worlds, 'candidates':candidates, 'reference_processed':ref}, args.output_dir / f'task_{task_id}_forecasts.pt')
        finally:
            env.close()
        real_replays, targets, real_images = [], [], []
        for candidate in candidates:
            replay_env, replay_obs = r.make_env(task, 'libero_10', args.seed)
            try:
                if digest(replay_env.env.sim.get_state().flatten()) != initial_sim_hash:
                    raise RuntimeError('initial simulator state changed across condition replays')
                result = r.replay_once(replay_env, replay_obs, candidate[:args.replay_horizon])
                after = result.pop('observations_after_steps')[3]
                image = canonical_image(after)
                real_images.append(image)
                targets.append(r.patchify(model.image_processor(image).unsqueeze(0))[0])
                real_replays.append(result)
            finally:
                replay_env.close()
        row['candidate_replays'] = real_replays
        target = torch.stack(targets)
        row['candidate_forecast_mse'] = {name: ((pred-target)**2).mean((1,2)).tolist() for name,pred in worlds.items()}
        row['evaluation_only_real_image_clip_scores'] = visual_scores(model, real_images, task.language).tolist()
        torch.save({'target_patches_step3':target}, args.output_dir / f'task_{task_id}_targets.pt')
        rows.append(row)
        result = {**protocol, 'status': 'running', 'rows': rows}
        (args.output_dir / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'task_id':task_id, 'selected':row['selected'],
                          'real_eef': [v['eef_approach_progress'] for v in real_replays]}), flush=True)
    result['status'] = 'completed'
    (args.output_dir / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
