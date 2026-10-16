#!/usr/bin/env python3
"""Summarize the corrected fixed-action forecast gate, not task attack success."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def flatten(value):
    if isinstance(value, list):
        return [x for part in value for x in flatten(part)]
    return [value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-audit', type=Path,
                        help='Parsed BDDL compatibility audit; derive matched goals from metadata only')
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    if data['status'] != 'completed_prediction_contract_gate':
        parser.error('require a completed prediction contract gate')
    clean = data['goal_ids'].index(data['clean_goal_id'])
    matched_ids = []
    scene_audit = None
    if args.scene_audit:
        scene_audit = json.loads(args.scene_audit.read_text())
        contracts = {row['task_id']: row['physical_contract_sha256']
                     for row in scene_audit['tasks']}
        matched_ids = [goal for goal in data['goal_ids']
                       if goal != data['clean_goal_id']
                       and contracts[goal] == contracts[data['clean_goal_id']]]
        if not matched_ids:
            parser.error('no alternative goal with the same physical BDDL contract')
    matched_indices = [data['goal_ids'].index(goal) for goal in matched_ids]
    chunks = []
    for row in data['rows']:
        error = row['target_mse_by_goal_action_view']
        if not all(r['initial_exact'] and r['future_exact'] for r in row['replays']):
            raise ValueError('physical replay pairing failed')
        if not row['forecast_repeat_exact']:
            raise ValueError('fixed-input forecast repeat failed')
        chunks.append({
            'chunk_index': row['chunk_index'],
            'Clean': mean(flatten(error['native'][clean])),
            'GoalSwapUniform': mean(flatten([v for i, v in enumerate(error['native']) if i != clean])),
            'CleanBusCut': mean(flatten(error['bus_cut'][clean])),
            'GoalSwapBusCut': mean(flatten([v for i, v in enumerate(error['bus_cut']) if i != clean])),
            'GoalSwapControlCut': mean(flatten([v for i, v in enumerate(error['control_cut']) if i != clean])),
            'Persistence': row['persistence_target_mse_mean'],
        })
        if matched_indices:
            for name, mask in [('SameSceneGoalSwap', 'native'),
                               ('SameSceneGoalSwapBusCut', 'bus_cut'),
                               ('SameSceneGoalSwapControlCut', 'control_cut')]:
                chunks[-1][name] = mean(flatten([error[mask][i] for i in matched_indices]))
    averages = {key: mean(row[key] for row in chunks) for key in chunks[0] if key != 'chunk_index'}
    result = {
        'status': 'completed_corrected_forecast_analysis',
        'source': str(args.summary),
        'source_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        'episode_clusters': 1,
        'transitions': sum(len(row['candidate_names']) for row in data['rows']),
        'replays': sum(len(row['replays']) for row in data['rows']),
        'mean_native_rgb_patch_mse': averages,
        'per_chunk': chunks,
        'swap_minus_clean': averages['GoalSwapUniform'] - averages['Clean'],
        'goal_scope': {
            'GoalSwapUniform': 'Uniform over all other corpus task texts; most specify different physical scenes. Exploratory cross-scene sensitivity, not uniformly same-scene legal goals.',
            'same_scene_goal_ids': matched_ids,
            'same_scene_selection': 'Physical BDDL metadata equality excluding goal predicates, instruction and objects of interest; not chosen using errors.',
            'same_scene_swap_minus_clean': (averages['SameSceneGoalSwap'] - averages['Clean']) if matched_indices else None,
            'scene_audit_source': str(args.scene_audit) if args.scene_audit else None,
            'scene_audit_sha256': hashlib.sha256(args.scene_audit.read_bytes()).hexdigest() if args.scene_audit else None,
            'same_scene_does_not_prove_training_observation_sufficiency': True,
        },
        'bus_minus_clean': averages['CleanBusCut'] - averages['Clean'],
        'bus_goal_exact_all': all(r['bus_cut_cross_goal_exact'] for r in data['rows']),
        'control_world_exact_all': all(r['control_cut_native_forecast_exact'] for r in data['rows']),
        'bus_action_sensitive_all': all(
            value > 0 for row in data['rows']
            for value in flatten(row['action_sensitivity_mse_by_goal_view']['bus_cut'])
        ),
        'uncertainty': 'No CI or significance claim: correlated chunks of one development episode. Goal assignments are crossed; training-distribution observation sufficiency is not established.',
        'task_attack_or_defense_success_tested': False,
        'environment_steps_new_for_analysis': 0,
        'api_requests': 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
