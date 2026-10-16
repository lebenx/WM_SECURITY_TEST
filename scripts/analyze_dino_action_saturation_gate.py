#!/usr/bin/env python3
"""Summarize the fixed DINO alias gate from recorded real outcomes; no inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/dino_action_saturation_gate_20261015')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/dino_action_saturation_decision_20261015.json')
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    protocol = json.loads((run / 'launched_config.json').read_text())
    assert summary['status'] == 'completed_cpu_fixed_bank_gate'
    assert summary['snapshot']['historical_source_prefix_verified']
    names = summary['bank']['candidate_names']
    original = summary['forecasts']['original']
    rows = []
    for i, name in enumerate(names):
        trace = [json.loads(line) for line in (run / f'candidate{i}/original/trace.jsonl').read_text().splitlines()]
        assert len(trace) == 30
        metrics = summary['utilities'][i]
        for horizon, key in [(15, 'native_execution_prefix15'), (30, 'primary30')]:
            actual = metrics[key]
            assert actual['terminal_native_reward'] == trace[horizon - 1]['reward']
            assert actual['summed_native_reward'] == sum(row['reward'] for row in trace[:horizon])
            assert actual['final_reach_distance'] == trace[horizon - 1]['reach_distance']
            assert actual['final_success'] == trace[horizon - 1]['success']
        row = dict(candidate=i, name=name, clean_native_cost=original['native_costs'][i],
                   clean_predicted_rank=original['ranking'].index(i) + 1,
                   clean_selected=i == original['selected_candidate'])
        for suffix, key in [('30', 'primary30'), ('15', 'native_execution_prefix15')]:
            row.update({f'{k}_{suffix}': v for k, v in metrics[key].items() if k != 'controls'})
        row['clean_final_weighted_feature_forecast_error'] = original['forecast_error']['weighted']['final_per_candidate'][i]
        rows.append(row)
    table = run / 'candidate_results.csv'
    with table.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    selected = original['selected_candidate']
    best30 = max(rows, key=lambda row: row['terminal_native_reward_30'])
    best15 = max(rows, key=lambda row: row['terminal_native_reward_15'])
    chosen = rows[selected]
    alias_pairs = [x for x in summary['replays'] if x['representation'] != 'original']
    assert len(summary['replays']) == 15 and len(alias_pairs) == 10
    assert all(x['controls'] == 30 and x['exact_to_original'] for x in alias_pairs)
    gate = summary['gate']
    # This analysis deliberately does not relax the preregistered oracle gate.
    assert gate['clean_competence_passed'] is False
    assert set(summary['forecasts']) == {'original'}
    assert all(gate[k] is None for k in ('harm_signal_conditioned_on_clean', 'canonical_recovery_if_harm', 'inflated_selected_index_changed'))
    report = {
        'schema_version': 1,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_physical_equivalence_clean_oracle_gate_failed_alias_forecasts_unmeasured',
        'previous_goal_turn_classification': 'progress: completed strong-CEM paired task experiment and synchronized archive; real return tie limits its decision-harm claim',
        'research_goal_status': 'active_unachieved',
        'question': 'Can execution-equivalent saturated XYZ representations change pretrained DINO forecasts and real selected-action utility?',
        'protocol': protocol,
        'physical_evidence': {
            'independent_root_states': 1, 'fixed_candidates': 5, 'representations_per_candidate': 3,
            'replay_branches': 15, 'explicit_environment_controls': 450,
            'alias_vs_original_paired_branches': 10, 'paired_step_comparisons': 300,
            'historical_source_prefix_controls_exact': 30,
            'exact_checks': ['effective clipped XYZ and unchanged gripper', 'per-step full mjSTATE_INTEGRATION', 'recorded state and TCP', 'rendered pixels', 'native reward and success/termination', 'restored counters and observation history'],
            'snapshot': summary['snapshot'], 'action_construction': summary['bank'],
            'scope': 'One fixed reach root with equal committed controls. The environment constructor may simulate internally;450 counts explicit step calls only. Branches are related, not15 independent tasks.'
        },
        'candidate_results': rows,
        'clean_decision': {
            'selected_candidate': selected, 'oracle_best30_candidate': best30['candidate'],
            'selected_terminal_reward30': chosen['terminal_native_reward_30'],
            'oracle_terminal_reward30': best30['terminal_native_reward_30'],
            'terminal_reward30_regret': best30['terminal_native_reward_30'] - chosen['terminal_native_reward_30'],
            'regret_fraction_of_best30_reward': (best30['terminal_native_reward_30'] - chosen['terminal_native_reward_30']) / best30['terminal_native_reward_30'],
            'selected_beats_both_constructed_controls': all(chosen['terminal_native_reward_30'] > rows[i]['terminal_native_reward_30'] for i in (3, 4)),
            'oracle_best15_candidate': best15['candidate'],
            'selected_is_best15': selected == best15['candidate'],
            'interpretation': 'Clean selects the second-best30-control candidate, near the oracle and above both controls. It is also best by the predeclared secondary15-control terminal reward. This does not pass the frozen30-control oracle-optimality gate;15-control results do not replace the primary endpoint. Gate failure is not evidence of general model/controller incompetence.'
        },
        'gate': gate,
        'statistics': {'independent_roots': 1, 'significance_tests': None, 'confidence_intervals': None,
                       'reason': 'One deliberately constructed development bank; no independent statistical sample or population claim.'},
        'scientific_decision': {
            'physical_equivalence': 'Go for this exact finite action bank and reach root: paired future states, images and utilities are equal.',
            'current_decision_harm_claim': 'No-Go: frozen Clean oracle gate failed, so intervention/canonical forecasts were not evaluated.',
            'not_a_negative_attack_result': 'Unmeasured alias ranking/harm/recovery are null, not zero. This run does not show that aliases are harmless or canonicalization ineffective.',
            'native_scope': 'Released pretrained DINO-WM model/unroll and native objective, but an added fixed open-loop selector. Full native CEM uses15-step execution before replanning; no CEM or whole task episode was run here.',
            'conditional_prediction_limit': 'Same real state and executed control establishes paired intervention equivalence, not universal alias invariance of observational Bayes P(Y|O,A) under partial observability and behavior-policy confounding.',
            'threat_model_limit': 'A forecast-request-only action representation writer is a proposed restricted permission. The inspected system directly calls local unroll; no deployed low-trust writer was identified.',
            'novelty': 'No numerical novelty rating is justified. Clipping equivalence and model exploitation are established prior art; even a later positive fixed-bank result would not close native-harm, deployed-interface or generalization gaps.',
            'next_action': 'Preserve the exact alias/replay harness. Do not retune this bank, magnitude, metric or gate using its results. A later alias study requires a concrete representation-writer contract and a separately frozen native execution protocol; oracle optimality is a stricter gate than ordinary competence and is not a universal safety prerequisite. Without that new basis, screen a materially different native decision interface before another task run.'
        },
        'activity': {**summary['activity'], 'checkpoint_loads': summary['checkpoint_loads'],
                     'elapsed_seconds': summary['elapsed_seconds'], 'original_unroll_seconds': original['seconds'],
                     'gpu_initialized': summary['cuda_initialized'], 'model_api_requests': 0,
                     'new_training': 0, 'new_checkpoint_downloads': 0, 'new_large_weight_hashes': 0},
        'primary_sources': [
            {'title': 'Clipped Action Policy Gradient', 'url': 'https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf', 'checked_scope': 'section3 equations4–6 and section3.4', 'relevance': 'Prior clipping-equivalence assumptions and raw-action reward exception; not our experiment.'},
            {'title': 'World Models', 'url': 'https://worldmodels.github.io/#cheating-the-world-model', 'checked_scope': 'Cheating the World Model section', 'relevance': 'Prior learned-model exploitation; not our experiment.'},
            {'title': 'When to Trust Your Model: Model-Based Policy Optimization', 'url': 'https://arxiv.org/html/1906.08253', 'checked_scope': 'sections4 and6', 'relevance': 'Prior model-error/distribution-shift and exploitation analysis; not our experiment.'}
        ],
        'artifacts': {'run': str(run.relative_to(ROOT)), 'summary': str((run / 'summary.json').relative_to(ROOT)),
                      'candidate_table': str(table.relative_to(ROOT)), 'frozen_bank': str((run / 'frozen_bank.npz').relative_to(ROOT)),
                      'forecasts_and_truth': str((run / 'forecasts_and_encodings.npz').relative_to(ROOT)),
                      'runner': 'scripts/probe_dino_action_saturation_gate.py', 'analysis': str(Path(__file__).resolve().relative_to(ROOT))},
        'reproduction': {'experiment': '.venv-jepa-native/bin/python scripts/probe_dino_action_saturation_gate.py --out /absolute/path/to/new_output',
                         'analysis': 'python scripts/analyze_dino_action_saturation_gate.py',
                         'note': 'Experiment refuses existing output directories. The analysis reads cached traces/summary only and makes no model/environment/API calls.'}
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.report), 'table': str(table), 'status': report['status'], 'clean_decision': report['clean_decision']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
