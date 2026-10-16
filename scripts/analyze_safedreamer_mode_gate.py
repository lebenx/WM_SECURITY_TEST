#!/usr/bin/env python3
"""Summarize the frozen SafeDreamer gate from completed execution records.

No checkpoint, neural model, environment, network, or new scientific run is used.
The separate independent review checks cached tensors rather than these summaries.
"""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/safedreamer_mode_mediator_gate_20261016'
REPORT = ROOT / 'reports/safedreamer_mode_mediator_gate_20261016.json'


def read(path):
    return json.loads(path.read_text())


def main():
    report = read(RUN / 'report_before_scientific_execution.json')
    native = read(RUN / 'native_query_attempt1/summary.json')
    replay = read(RUN / 'replay_attempt1/summary.json')
    setup = read(RUN / 'environment_continuation_20261016/verification.json')
    assert native['status'] == 'completed_native_first_query_preflight'
    assert replay['status'] == 'completed_no_mode_switch_stop'
    assert replay['calibration_passed_before_common'] and all(replay['checks'].values())
    assert not replay['mode_gate_passed']
    arms = replay['arms']
    assert all(not arm['selected_padding'] for arm in arms.values())
    assert all(arm['safe_count'] >= 50 for arm in arms.values())
    assert native['activity']['environment_steps'] == replay['activity']['environment_steps'] == 0
    action_distance = math.sqrt(sum((x-y)**2 for x, y in zip(
        arms['independent']['returned_action'], arms['common']['returned_action'])))
    report.update(
        status='completed_fixed_native_mode_gate_no_crossing_stop',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress_actual_pretrained_inference_and_fixed_gate_result',
        next_action='Stop this fixed operation: no replacement query, seed, root, threshold, horizon or coupling search. Retain the qualified runtime and caches. A future experiment requires a substantively different, justified mechanism and real-utility evidence plan; no next scientific run is currently qualified.',
    )
    report['historical_setup_report'] = str((RUN / 'report_before_scientific_execution.json').relative_to(ROOT))
    report['environment_continuation'] = {
        'verification': str((RUN / 'environment_continuation_20261016/verification.json').relative_to(ROOT)),
        'resource_amendment': str((RUN / 'environment_continuation_20261016/resource_amendment.json').relative_to(ROOT)),
        'official_archive_bytes': 578448090,
        'resource_seconds': setup['total_resource_seconds'],
        'status': setup['status'],
        'scope': setup['dependency_upgrade_scope'],
        'original_setup_stop_preserved': True,
    }
    report['resources']['checkpoint_deserialized'] = True
    report['resources']['isolated_environment'] = '.venv-safedreamer'
    report['resources']['environment_camera_rendering_now_verified'] = True
    report['resources']['gpu_native_inference_now_verified'] = True
    report['resources']['training_environment_identity_known'] = False
    report['activity'].update({
        'checkpoint_loads': 2, 'model_constructions': 2,
        'model_forward_calls': None,
        'forward_counting_rule': 'Report native CEM rounds and explicit final-round replays below; there is no instrumented total of all neural submodule calls.',
        'native_policy_queries': 1, 'native_policy_queries_completed': 1,
        'native_cem_rounds': 6, 'final_round_replays': 2,
        'replay_rssm_batch_steps': 30, 'imagined_candidate_transitions': 60000,
        'environment_constructions': 1, 'environment_resets': 1,
        'environment_steps': 0, 'full_clean_episodes': 0,
        'training_updates': 0, 'model_api_calls': 0, 'gpu_model_runs': 2,
        'official_environment_archives_downloaded': 1,
    })
    report['execution'] = {
        'native_summary': str((RUN / 'native_query_attempt1/summary.json').relative_to(ROOT)),
        'replay_summary': str((RUN / 'replay_attempt1/summary.json').relative_to(ROOT)),
        'sessions': str((RUN / 'execution_sessions.json').relative_to(ROOT)),
        'native_total_seconds': native['elapsed_seconds'],
        'native_query_seconds': native['query_wall_seconds'],
        'native_peak_gpu_mib': native['peak_process_gpu_mib'],
        'replays_total_seconds': replay['elapsed_seconds'],
        'replay_seconds': replay['replay_seconds'],
        'replays_peak_gpu_mib': replay['peak_process_gpu_mib'],
        'runtime_checks_passed': len(replay['checks']),
        'full_independent_calibration_exact': True,
    }
    report['scientific_gate'] = {
        'status': 'measured_no_mode_crossing_fixed_operation_stopped',
        'native_safe_count': arms['independent']['safe_count'],
        'shared_safe_count': arms['common']['safe_count'],
        'threshold': 50, 'candidates': 500,
        'native_mode': arms['independent']['mode'], 'shared_mode': arms['common']['mode'],
        'padding_selected': False, 'mode_crossing': False,
        'returned_action_l2_difference': action_distance,
        'arms': arms,
        'forecast_error_against_environment': None,
        'action_ranking_flip_rate': None,
        'actual_selected_action_utility': None,
        'full_clean_capability': None, 'actual_harm': None, 'defense_recovery': None,
        'reason': 'Exact calibrated fixed intervention changed predicted safe count and returned action, but did not cross the native switch. The frozen prerequisite for full Clean and physical continuation failed; neither was run.',
        'same_forecast_control_scope': 'Both always-minimum-cost actions were computed from the existing forecast tensors without new model calls. They were not executed and are not a demonstrated defense.',
    }
    report['statistics'] = {
        'independent_task_replications': 0, 'preselected_development_queries': 1,
        'descriptive_only': True, 'population_asr': None, 'p_value': None,
        'rule': '500 CEM alternatives are not 500 independent task trials. An elite-mean action change is not a measured action-ranking flip rate or real harm.',
    }
    report['implementation']['first_query_validation'] = 'Native GPU query completed, strict inference-key checks and first actual seed0 reset succeeded.'
    report['implementation']['replay_validation'] = 'Independent static review plus actual exact independent calibration, protected-input checks, and one completed common-key replay.'
    report['implementation']['replay_static_review'] = 'reports/safedreamer_mode_replay_review_20261016.json'
    report['implementation']['analysis'] = str(Path(__file__).relative_to(ROOT))
    report['implementation']['table'] = str(REPORT.with_suffix('.csv').relative_to(ROOT))
    review = ROOT / 'reports/safedreamer_mode_actual_result_review_20261016.json'
    if review.exists():
        report['implementation']['independent_actual_result_review'] = str(review.relative_to(ROOT))
    report['decision'] = {
        'go': 'Retain successfully loaded official pretrained model, exact native replay and pinned rendering runtime as reusable assets.',
        'no_go': 'Do not advance this fixed safety-mode intervention to a full Clean/attack/defense panel or claim decision-level harm. No mode crossing; no physical utility result.',
        'scope': 'One development query and one fixed shared-stream operation; not a proof that all possible dependence interventions are harmless.',
        'global_goal_complete': False, 'qualified_core_security_claim': None,
        'external_user_action_required': False, 'exclusive_gpu_justified': False,
        'live_processes': False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with REPORT.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sampling', 'decision_rule', 'safe_count', 'mode',
            'action_0', 'action_1', 'selected_padding', 'physical_action_executed', 'actual_utility'])
        writer.writeheader()
        for sampling, arm in arms.items():
            for decision in ('native', 'always_minimum_cost'):
                action = arm['returned_action' if decision == 'native' else 'always_minimum_cost_action']
                writer.writerow(dict(sampling=sampling, decision_rule=decision, safe_count=arm['safe_count'],
                    mode=arm['mode'] if decision == 'native' else 'minimum_cost', action_0=action[0],
                    action_1=action[1], selected_padding=False, physical_action_executed=False,
                    actual_utility='unmeasured'))
    print(json.dumps({'report': str(REPORT.relative_to(ROOT)), 'status': report['status'],
        'safe_counts': [arms[x]['safe_count'] for x in ('independent', 'common')],
        'mode_crossing': False, 'physical_steps': 0, 'action_l2_difference': action_distance}))


if __name__ == '__main__':
    main()
