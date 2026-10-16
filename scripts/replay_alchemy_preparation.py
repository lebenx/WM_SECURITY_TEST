"""Replay matched native checkpoints to isolate existing planning capability."""
from collections import Counter
from copy import copy
import json
from pathlib import Path
import time

import screen_alchemy_preparation as screen

np, sp, io, sa = screen.np, screen.sp, screen.io, screen.sa
trackers, utils = screen.trackers, screen.utils


def run(seed, checkpoint, prefix, policy, pre):
    env = sa.get_symbolic_alchemy_level(
        'all_fixed_with_random_bottleneck', seed=seed, num_trials=4,
        num_stones_per_trial=1, num_potions_per_trial=4,
        max_steps_per_trial=8, end_trial_action=True)
    tracker = trackers.BeliefStateTracker(pre, env)
    env.add_trackers({tracker.NAME: tracker})
    counts = Counter()
    phase = 'replay'
    native_step = env.step_no_observation

    def counted_step(action, *args, **kwargs):
        counts[phase] += 1
        return native_step(action, *args, **kwargs)

    env.step_no_observation = counted_step
    env.reset()
    for saved in prefix:
        belief = tracker.belief_state.belief_state
        assert screen.remaining_worlds(belief) == saved['before_worlds']
        assert {str(k): int(v) for k, v in belief.aligned_stones.items()} == saved['stones']
        assert {str(k): int(v) for k, v in belief.perceived_potions.items()} == saved['potions']
        action = utils.type_based_action_from_ints(
            *saved['selected_action'], tracker.belief_state.rotation)
        sa.take_simplified_action(action, env)
        assert screen.remaining_worlds(tracker.belief_state.belief_state) == saved['after_worlds']
    belief = tracker.belief_state.belief_state
    assert {str(k): int(v) for k, v in belief.aligned_stones.items()} == checkpoint['stones']
    assert {str(k): int(v) for k, v in belief.perceived_potions.items()} == checkpoint['potions']
    before = screen.remaining_worlds(belief)
    assert before == checkpoint['before_worlds']
    trial = env.trial_number
    assert trial == checkpoint['trial']
    phase = 'continuation'
    cache = {}
    trajectory = []
    while True:
        if counts[phase] > 8:
            raise RuntimeError('Continuation exceeded the four-potion trial bound')
        belief = tracker.belief_state.belief_state
        objectives = screen.one_step_objectives(belief, pre)
        if policy == 'native_ideal_explorer':
            action, objective, cache = io.ideal_observer(belief, cache, 12, pre, True)
        else:
            chosen = min(objectives, key=lambda x: round(x['expected_remaining_worlds'], 10))
            action = tuple(chosen['action'])
            objective = -chosen['expected_remaining_worlds']
        step = {'before_worlds': screen.remaining_worlds(belief),
                'action': [int(v) for v in action], 'objective': float(objective)}
        timestep = sa.take_simplified_action(utils.type_based_action_from_ints(
            *action, tracker.belief_state.rotation), env)
        step['after_worlds'] = screen.remaining_worlds(tracker.belief_state.belief_state)
        trajectory.append(step)
        if timestep.last() or env.trial_number != trial:
            break
    return {'seed': seed, 'checkpoint_decision': len(prefix), 'trial': trial,
            'policy': policy, 'initial_worlds': before,
            'final_worlds': trajectory[-1]['after_worlds'],
            'native_actions': dict(counts), 'continuation': trajectory}


def main():
    source = Path('runs/alchemy_preparation_baselines_v1')
    output = Path('runs/alchemy_preparation_replay_v0')
    output.mkdir(exist_ok=False)
    save = lambda name, x: (output / name).write_text(json.dumps(x, indent=2) + '\n')
    source_summary = json.loads((source / 'summary.json').read_text())
    checkpoints = [d for d in source_summary['zero_immediate_information_preparation_decisions']
                   if abs(d['best_expected_immediate_reduction']) < 1e-9]
    save('plan.json', {'source': str(source), 'selection': 'All exploratory checkpoints with zero best immediate information reduction and positive native expected continuation reduction.',
                      'checkpoint_count': len(checkpoints), 'independent_holdout': False,
                      'scope': 'Matched-history intervention on existing target-only planning, not transfer or a new method.',
                      'api_requests': 0, 'gpu_seconds': 0})
    start = time.monotonic()
    pre = copy(screen.precomputed_maps.load_from_level_name(
        'perceptual_mapping_randomized_with_random_bottleneck'))
    pre.stone_maps = np.array([sp.all_fixed_stone_map().index()])
    pre.potion_maps = np.array([sp.all_fixed_potion_map().index(pre.perm_index_to_index)])
    results = []
    for checkpoint in checkpoints:
        seed = checkpoint['seed']
        episode = json.loads((source / f'seed{seed}_native_ideal_explorer.json').read_text())
        prefix = episode['decisions'][:checkpoint['decision']]
        for policy in ('myopic_world_count', 'native_ideal_explorer'):
            result = run(seed, checkpoint, prefix, policy, pre)
            results.append(result)
            save(f'seed{seed}_{policy}.json', result)
            print(json.dumps(result), flush=True)
    summary = {'status': 'completed', 'matched_checkpoints': len(checkpoints),
               'native_actions': sum(sum(r['native_actions'].values()) for r in results),
               'native_replay_actions': sum(r['native_actions']['replay'] for r in results),
               'native_continuation_actions': sum(r['native_actions']['continuation'] for r in results),
               'results': results, 'api_requests': 0, 'gpu_seconds': 0,
               'elapsed_s': time.monotonic() - start,
               'interpretation': 'Existing native within-trial exhaustive planning tested against one-step selection from identical replayed histories. Selected diagnostic checkpoints, not independent held-out tasks. No novel method or cross-world experiment transfer evaluated.'}
    save('summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
