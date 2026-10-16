#!/usr/bin/env python3
"""Publish cached DINO alias contrasts without running a model or environment."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/dino_action_saturation_exploratory_20261015')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/dino_action_alias_exploratory_decision_20261015.json')
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    prior = json.loads((ROOT / 'runs/dino_action_saturation_gate_20261015/summary.json').read_text())
    amendment = json.loads((ROOT / 'reports/dino_action_alias_gate_amendment_20261015.json').read_text())
    oracle = json.loads((run / 'oracle_objective_analysis.json').read_text())
    assert summary['status'] == 'completed_exploratory_cached_alias_forecasts'
    assert summary['runtime_replay']['exact']
    assert summary['prior_oracle_gate_passed'] is False
    assert prior['gate']['clean_competence_passed'] is False
    assert summary['activity']['environment_steps'] == summary['activity']['native_encodes'] == 0
    names = summary['candidate_names']
    condition_rows, candidate_rows = [], []
    for name, condition in summary['conditions'].items():
        selected = condition['selected_candidate']
        utility = condition['selected_real_utility']
        assert utility == prior['utilities'][selected]
        errors = condition['forecast_errors']['weighted']['final_per_candidate']
        row = dict(condition=name, selected_candidate=selected, selected_name=names[selected],
                   rank_order=' '.join(map(str, condition['ranking'])),
                   mean_final_weighted_feature_error=sum(errors) / len(errors))
        for endpoint, suffix in [('primary30', '30'), ('native_execution_prefix15', '15')]:
            for key, value in utility[endpoint].items():
                if key != 'controls':
                    row[f'{key}_{suffix}'] = value
        condition_rows.append(row)
        for candidate, label in enumerate(names):
            candidate_rows.append(dict(condition=name, candidate=candidate, name=label,
                native_goal_cost=condition['native_costs'][candidate],
                predicted_rank=condition['ranking'].index(candidate) + 1,
                final_weighted_feature_error=errors[candidate],
                terminal_real_reward30=prior['utilities'][candidate]['primary30']['terminal_native_reward'],
                selected=candidate == selected))
    write_csv(run / 'condition_results.csv', condition_rows)
    write_csv(run / 'candidate_forecast_results.csv', candidate_rows)
    original, inflated, canonical = condition_rows
    assert original['condition'] == 'original' and inflated['condition'] == 'inflated_saturated_xyz'
    assert canonical['condition'] == 'canonical_saturated_xyz'
    contrasts = summary['signed_descriptive_contrasts']
    fixed = contrasts['inflated_saturated_xyz_minus_original']
    repair = contrasts['canonical_saturated_xyz_minus_original']
    assert fixed['primary30']['terminal_native_reward'] == inflated['terminal_native_reward_30'] - original['terminal_native_reward_30']
    assert repair['primary30']['terminal_native_reward'] == canonical['terminal_native_reward_30'] - original['terminal_native_reward_30']
    no_primary_loss = inflated['terminal_native_reward_30'] >= original['terminal_native_reward_30'] - 1e-6
    assert no_primary_loss, 'This report is for the observed no-primary-harm stopping branch.'
    true_scoring = oracle['true_future_native_objective']
    assert true_scoring['selected_candidate'] == canonical['selected_candidate']
    assert true_scoring['selected_real_utility']['primary30']['terminal_native_reward'] == canonical['terminal_native_reward_30']
    result = {
        'schema_version': 1,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_exploratory_fixed_inflation_no_harm_canonicalization_utility_loss',
        'previous_goal_turn_classification': 'progress: physical equivalence and Clean predictions measured; oracle gate halted the still-unobserved representation effect',
        'research_goal_status': 'active_unachieved',
        'methodological_amendment': amendment,
        'protocol': 'configs/dino_action_saturation_exploratory_20261015.json',
        'source_physics': {'run': 'runs/dino_action_saturation_gate_20261015',
            'independent_roots': 1, 'candidates': 5, 'prior_replay_branches': 15,
            'prior_explicit_controls': 450, 'new_explicit_controls_this_continuation': 0,
            'scope': 'Cached real environment consequences, with exact paired execution equivalence established previously; not predicted utilities or new task episodes.'},
        'condition_results': condition_rows,
        'signed_contrasts': contrasts,
        'canonicalization_effect': {
            'terminal_real_reward_drop': original['terminal_native_reward_30'] - canonical['terminal_native_reward_30'],
            'terminal_real_reward_relative_drop': 1 - canonical['terminal_native_reward_30'] / original['terminal_native_reward_30'],
            'mean_final_feature_error_relative_reduction': 1 - canonical['mean_final_weighted_feature_error'] / original['mean_final_weighted_feature_error'],
            'interpretation': 'The proposed canonical repair reduces forecast feature error but changes selection to a lower-utility candidate. Report this observed repair side effect; do not retrospectively rename it the prespecified successful attack or call it recovery.'},
        'true_future_objective_control': oracle,
        'decisive_control_interpretation': {
            'finding': 'The unchanged native goal objective, supplied with the actual future encodings, selects the same low-utility candidate as canonicalization.',
            'selected_candidate': true_scoring['selected_candidate'],
            'selected_real_terminal_reward': true_scoring['selected_real_utility']['primary30']['terminal_native_reward'],
            'cost_margin_over_runner_up': true_scoring['best_vs_runner_up_cost_margin'],
            'exact_minimum_ties': true_scoring['exact_minimum_ties'],
            'implication': 'A perfect forecast of the features used by this objective would still choose the low-utility candidate. Canonicalization agrees with that choice. Original forecast score errors happen to favor a higher-real-utility candidate in this bank.',
            'causal_limit': 'The incremental representation-to-forecast-to-selection utility loss remains real. Objective/reward misalignment and forecast-mediated damage can coexist; the true-future control limits a unique attribution to newly introduced prediction error, not existence of the local effect. It does not quantify a general decomposition or require oracle-optimal scoring as a universal prerequisite.'},
        'scientific_decision': {
            'fixed_inflation': 'No-Go for expansion: +2 leaves the entire ranking, selected candidate and real utility unchanged. This is now a measured negative result for this fixed operation, not a claim that all aliases are harmless.',
            'canonical_repair': 'No-Go as a demonstrated defense: it lowers selected real utility on both predeclared horizons. No primary loss from +2 existed to recover.',
            'local_causal_evidence': 'On this fixed bank, a physically equivalent forecast-input canonicalization changes the selected candidate and its independently replayed real utility. The same low-utility candidate is selected by scoring the actual future encodings, exposing a native objective/reward mismatch. Preserve the local representation effect without promoting it to a forecast-contamination security mechanism or full native CEM task damage.',
            'native_scope': 'Released pretrained DINO-WM plus native unroll/objective, but an added30-control open-loop selector. Native CEM replans after15controls. All five cached30control outcomes have success=false.',
            'baseline': 'The old oracle-optimality gate remains failed. This separately recorded exploratory continuation uses a near-optimal local Clean baseline without retroactively installing a passing threshold.',
            'threat_model': 'Researcher-defined constrained forecast-request writer; no deployed external low-trust interface was discovered. Lack of deployment evidence limits deployment claims, not validity of the finite paired experiment.',
            'novelty': 'No novelty score or complete paper claim. Action-equivalence and model exploitation are established prior art; lower average feature error is not a guarantee of correct goal ranking.',
            'next_action': 'Stop the fixed+2 operation and canonicalization-as-defense promotion. Preserve the incremental forecast-mediated canonicalization effect alongside the true-future objective baseline; require neither a perfect Clean nor oracle-optimal goal scoring. This unexpected result can motivate a separately frozen hypothesis: whether execution-equivalent forecast preprocessing lowers native closed-loop utility while improving prediction quality. Before any larger run, specify the constrained writer and native execution path, select an independent panel without searching its outcomes, and check runtime cost. Keep prediction error, true-future proxy scoring and real utility separate. Do not retroactively call canonicalization the prespecified successful attack.'},
        'statistics': {'independent_roots': 1, 'candidate_pairs_per_rank_comparison': 10,
            'rank_counts_are_population_rates': False, 'significance_test': None, 'confidence_interval': None,
            'reason': 'One fixed development bank with dependent candidates and known Clean outcomes; no independent generalization or confirmation.'},
        'activity': {**summary['activity'], 'elapsed_seconds': summary['elapsed_seconds'],
            'checkpoint_loads': summary['checkpoint_loads'], 'cuda_initialized': summary['cuda_initialized'],
            'model_api_requests': 0, 'new_checkpoint_downloads': 0, 'new_large_weight_hashes': 0,
            'oracle_control_additional_model_calls': 0},
        'controls': {'original_full_forecast_and_cost_exact_to_cache': summary['runtime_replay'],
            'unchanged_zero_candidate_forecast_exact_all_conditions': all(c['zero_forecast_exact_to_original'] for c in summary['conditions'].values()),
            'canonical_clean_and_inflated_inputs': 'Prior frozen-bank construction verified exact identity; a single canonical batch represents both, not two independent defense trials.'},
        'artifacts': {'summary': str((run / 'summary.json').relative_to(ROOT)),
            'condition_table': str((run / 'condition_results.csv').relative_to(ROOT)),
            'candidate_table': str((run / 'candidate_forecast_results.csv').relative_to(ROOT)),
            'forecasts': str((run / 'condition_forecasts.npz').relative_to(ROOT)),
            'oracle_control': str((run / 'oracle_objective_analysis.json').relative_to(ROOT)),
            'oracle_analysis': 'scripts/analyze_dino_true_future_objective.py',
            'runner': 'scripts/probe_dino_action_saturation_exploratory.py',
            'analysis': 'scripts/analyze_dino_alias_exploratory.py'},
        'reproduction': {'experiment': '.venv-jepa-native/bin/python scripts/probe_dino_action_saturation_exploratory.py --out /absolute/path/to/new_output',
            'true_future_control': '.venv-jepa-native/bin/python scripts/analyze_dino_true_future_objective.py --out /absolute/path/to/new_oracle_report.json',
            'analysis': 'python scripts/analyze_dino_alias_exploratory.py',
            'note': 'Experiment refuses existing outputs and uses prior frozen bank/encodings/utilities. Analysis performs no inference or environment calls.'}
    }
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.report), 'condition_results': condition_rows,
                      'canonicalization_effect': result['canonicalization_effect']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
