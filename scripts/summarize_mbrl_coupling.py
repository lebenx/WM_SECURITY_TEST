"""Aggregate completed sampling experiments, retaining negative constructions."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    'mbrl_coupling_cartpole_v0',
    'mbrl_trajectory_coupling_cartpole_v1',
    'mbrl_rank_coupling_cartpole_v2',
    'mbrl_rank_coupling_cartpole_confirm_v2',
]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    training = read('runs/mbrl_coupling_cartpole_v0/model/summary.json')
    records = []
    plans_checked = 0
    for group in GROUPS:
        for path in sorted((ROOT / 'runs' / group).glob('*/plan.json')):
            plan = json.loads(path.read_text())
            for source, expected in plan['hashes'].items():
                assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source
            plans_checked += 1
        for path in sorted((ROOT / 'runs' / group).glob('*/summary.json')):
            result = json.loads(path.read_text())
            if result['status'] == 'trained':
                continue
            assert result['status'] == 'completed'
            assert result['model_hash'] == training['model_hash']
            for row in result['rows']:
                episode = json.loads((path.parent / f"episode_{row['seed']}.json").read_text())
                assert episode['summary'] == row
                assert len(episode['trajectory']) == row['steps']
                assert sum(t['reward'] for t in episode['trajectory']) == row['return_']
                assert row['model_hash_unchanged']
            records.append(dict(group=group, source=str(path.relative_to(ROOT)), **result))
    pairs = []
    for group in GROUPS[2:]:
        arms = {r['condition']: r for r in records if r['group'] == group}
        by_seed = {arm: {r['seed']: r for r in result['rows']} for arm, result in arms.items()}
        for seed, control in by_seed['independent'].items():
            targeted = by_seed['targeted'][seed]
            starts = []
            for arm in ['independent', 'targeted']:
                starts.append(read(f'runs/{group}/{arm}/episode_{seed}.json')['trajectory'][0]['obs'])
            assert starts[0] == starts[1]
            pairs.append(dict(stage='development' if group == GROUPS[2] else 'new_seed_confirmation',
                              seed=seed, independent_return=control['return_'],
                              targeted_return=targeted['return_'],
                              shared_return=by_seed.get('shared', {}).get(seed, {}).get('return_'),
                              return_drop=control['return_'] - targeted['return_'],
                              independent_source=f'runs/{group}/independent/episode_{seed}.json',
                              targeted_source=f'runs/{group}/targeted/episode_{seed}.json'))
    independent_mean = sum(r['independent_return'] for r in pairs) / len(pairs)
    targeted_mean = sum(r['targeted_return'] for r in pairs) / len(pairs)
    evaluation_steps = sum(r['native_environment_steps'] for r in records)
    usage = dict(api_requests=0, failed_api_requests=0, gpu_seconds=0,
                 native_evaluation_episodes=sum(len(r['rows']) for r in records),
                 native_evaluation_steps=evaluation_steps,
                 native_collection_steps=training['native_collection_steps'],
                 native_environment_steps=evaluation_steps + training['native_collection_steps'],
                 native_predictive_transitions=sum(r['predictive_transitions'] for r in records),
                 additional_surrogate_forward_transitions=sum(r.get('surrogate_forward_transitions', 0) for r in records),
                 surrogate_backward_work='Additional autodiff in the trajectory construction; no equivalent FLOP count recorded.',
                 neural_training_updates=training['gradient_updates'],
                 training_cpu_elapsed_seconds=training['training_seconds'],
                 summed_evaluation_process_elapsed_seconds=sum(r['elapsed_seconds'] for r in records),
                 timing_scope='Sum of per-process elapsed times, not total wall time or measured CPU utilization; some runs overlapped.')
    vendor = ROOT / 'vendor/mbrl-lib'
    pin = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'], text=True).strip()
    model_dir = ROOT / 'runs/mbrl_coupling_cartpole_v0/model'
    result = dict(status='completed_small_learned_model_security_gate',
                  updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  paired_rows=pairs, independent_mean=independent_mean, targeted_mean=targeted_mean,
                  relative_mean_return_drop=(independent_mean-targeted_mean)/independent_mean,
                  paired_declines=sum(r['return_drop'] > 0 for r in pairs),
                  training=training, runs=records, usage=usage,
                  verification=dict(frozen_plans_checked=plans_checked, all_recorded_source_hashes_match=True,
                                    all_episode_rewards_and_lengths_match=True,
                                    paired_initial_observations_match=True, model_state_hash=training['model_hash'],
                                    current_model_file_hashes={n: hashlib.sha256((model_dir/n).read_bytes()).hexdigest()
                                                              for n in ['model.pth', 'env_stats.pickle']},
                                    normalizer_scope='Loaded from one saved training artifact in all runs; current digest is not a separately recorded pre-run normalizer digest.',
                                    mbrl_pin=pin, vendor_worktree_clean=True),
                  limits=['One trained checkpoint, four evaluation seeds, one continuous-control task.',
                          'Two discovery and two subsequent confirmation seeds; pooled four-seed result is not an independent confirmation study.',
                          'K=8 model-generation budget matched per evaluation; phase optimization adds attacker work and episode lengths differ.',
                          'Independent control selects each candidate rank independently; native ensemble allocation can still couple underlying candidate predictions.',
                          'Both arms return one N=10 forecast aggregate; this is not an all-80-particle averaging baseline.',
                          'Native CEM and model classes, but reduced pilot settings; not full PETS benchmark reproduction.',
                          'Rank-law preservation concerns learned predictions, not accuracy against the physical environment.',
                          'Confirmation plans retain a stale stage text naming seeds 704/705; their arguments and episode records correctly specify 706/707.',
                          'The v0 antithetic label implements a half-cycle quantile shift, not Gaussian sign-reversal antithetic sampling.',
                          'No broad novelty or deployed sampler vulnerability claim.'])
    path = ROOT / 'runs/mbrl_rank_coupling_cartpole_v2/assessment.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['paired_rows', 'independent_mean', 'targeted_mean', 'paired_declines', 'usage', 'verification']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
