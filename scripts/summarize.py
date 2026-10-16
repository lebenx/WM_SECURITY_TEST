"""Build the development evidence summary directly from completed result files."""
from collections import defaultdict
import html
import json
from pathlib import Path
from wm_security.outcomes import move_outcome, MOVE_CODING_VERSION

ROOT=Path(__file__).resolve().parents[1]
rows=[]
def add_timer(folder,label):
    for p in (ROOT/folder).rglob('*_result.json'):
        d=json.loads(p.read_text())
        rows.append({'study':label,'model':d['model'],'arm':d['arm'],
                     'baseline_errors':sum(x['false_success'] for x in d['baseline']),
                     'post_errors':sum(x['false_success'] for x in d['evaluation']),
                     'probes':len(d['evaluation']),'new_mismatches':d['new_mismatches'],
                     'instances':1,'source':str(p.relative_to(ROOT))})

add_timer('runs/timer_pilot_v2','时间约束：一步预测')
add_timer('runs/timer_bootstrap_pilot','时间约束：真实经验初始化／API')
p=ROOT/'runs/timer_rollout_v2/results.json'
if p.exists():
    groups=defaultdict(list)
    for r in json.loads(p.read_text()):groups[r['arm']].append(r)
    clean=sum(x['false_success'] for x in groups.get('clean',[]))
    for arm,items in groups.items():
        rows.append({'study':'时间约束：递归预演','model':'Qwen2.5-Coder-7B','arm':arm,
                    'baseline_errors':clean,'post_errors':sum(x['false_success'] for x in items),
                    'probes':len(items),'instances':1,'source':str(p.relative_to(ROOT))})
for directory,label in [('charging_pilot','资源例外：WorldEvolver'),('charging_reflection_pilot','资源例外：WorldMind 提示适配')]:
    for p in (ROOT/'runs'/directory).glob('*_result.json'):
        d=json.loads(p.read_text())
        rows.append({'study':label,'model':'Qwen2.5-Coder-7B','arm':d['arm'],
                     'baseline_errors':sum(x['false_success'] for x in d['baseline']),
                     'post_errors':sum(x['false_success'] for x in d['evaluation']),
                     'probes':len(d['evaluation']),'instances':1,'source':str(p.relative_to(ROOT))})
for p in (ROOT/'runs/credit_pilot').glob('*_result.json'):
    d=json.loads(p.read_text());b=[x for x in d['baseline'] if x['step']==2]
    rows.append({'study':'无关错误扣分：跨设备规划','model':'Qwen2.5-Coder-7B','arm':d['arm'],
                  'baseline_errors':sum(not x['prediction']['product_done'] and x['actual']['product_done'] for x in b),
                  'post_errors':d['final_false_failure'],'probes':2,'instances':1,'source':str(p.relative_to(ROOT))})
for p in (ROOT/'runs/failure_curriculum_pilot').glob('*_result.json'):
    d=json.loads(p.read_text())
    rows.append({'study':'真实失败：错误不可行判断','model':'Qwen2.5-Coder-7B','arm':d['arm'],
                 'baseline_errors':sum(x['false_failure'] for x in d['baseline']),
                 'post_errors':sum(x['false_failure'] for x in d['evaluation']),
                 'probes':len(d['evaluation']),'instances':1,'source':str(p.relative_to(ROOT))})

native_runs=[]
for directory in ['scienceworld_curriculum_v0', 'scienceworld_curriculum_local', 'scienceworld_freezer_local']:
    folder=ROOT/'runs'/directory
    if not (folder/'config.json').exists():continue
    config=json.loads((folder/'config.json').read_text())
    for p in folder.glob('*_result.json'):
        d=json.loads(p.read_text())
        native_runs.append({'source':str(p.relative_to(ROOT)), 'arm':d['arm'], 'status':d['status']})
        if d['status']!='completed':continue
        for r in d['baseline']+d['evaluation']:
            r['coded_move']=move_outcome(r['prediction']['prediction'])
        rows.append({'study':'ScienceWorld：原生更新／放置前提预测', 'model':config['model'], 'arm':d['arm'],
                     'baseline_errors':sum(r['coded_move'] is False for r in d['baseline']),
                     'post_errors':sum(r['coded_move'] is False for r in d['evaluation']),
                     'unknown_predictions':sum(r['coded_move'] is None for r in d['evaluation']),
                     'coding_version':MOVE_CODING_VERSION,
                     'probes':len(d['evaluation']), 'instances':1, 'source':str(p.relative_to(ROOT)),
                     'metric_note':'Conservative text coding; raw predictions retained. Not a task success rate.'})

diagnostics={}
for label,filename in [
    ('qwen3_exception_gate','runs/erasure_qwen3_pilot/status.json'),
    ('true_persistence_rule','runs/true_rule_qwen3/gate.json'),
    ('api_memory_component_repeats','runs/memory_dependence_api/result.json'),
    ('native_memory_swaps','runs/scienceworld_causal_local/memory_swaps.json'),
    ('native_actor_decisions','runs/scienceworld_causal_local/actor_decisions.json')]:
    p=ROOT/filename
    if p.exists():
        d=json.loads(p.read_text())
        if isinstance(d,dict):d.pop('identical_prompt_groups',None)
        elif label=='native_memory_swaps':
            d=[{'condition':r['condition'],'id':r['id'],'coded_move':move_outcome(r['prediction']['prediction'])} for r in d]
        elif label=='native_actor_decisions':
            d=[{'condition':r['condition'],'success':r['success'], 'actions':[s['action'] for s in r['steps']],
                'forecasts':[s['forecast']['prediction'] if s['forecast'] else None for s in r['steps']]} for r in d]
        diagnostics[label]={'source':filename,'result':d}

capture=ROOT/'runs/scienceworld_capture_local'
if (capture/'recovery.json').exists():
    diagnostics['retrieval_capture_recovery']={
        'source':'runs/scienceworld_capture_local/recovery.json',
        'result':[{'corrective_moves':r['corrective_moves'],
                   'forecasts':[{'view':q['view'],'coded_move':move_outcome(q['prediction']['prediction']),
                                 'retrieved_indices':q['prediction']['retrieved_indices']}
                                for q in r['evaluation']]}
                  for r in json.loads((capture/'recovery.json').read_text())]}
for label,folder in [('glm_memory_transfer','scienceworld_causal_glm'),
                     ('local_freeze_continuation','freeze_continuation_local'),
                     ('glm_actor_freeze_continuation','freeze_continuation_glm_actor'),
                     ('glm_full_freeze_continuation','freeze_continuation_glm_full'),
                     ('grounded_freeze_continuation','freeze_continuation_grounded_v0'),
                     ('matched_freeze_continuation_v0','freeze_continuation_matched_v0'),
                     ('matched_freeze_continuation_v1','freeze_continuation_matched_v1'),
                     ('glm_rule_dependence_gate','erasure_glm_gate'),
                     ('native_core_gold_replays','native_core_records_v0'),
                     ('paint_curriculum','paint_curriculum_local_v0'),
                     ('conductivity_curriculum','conductivity_glm_v0'),
                     ('conductivity_glm_repeats','conductivity_reads_glm'),
                     ('conductivity_glm_actor','conductivity_continuation_glm_v0'),
                     ('conductivity_local_actor','conductivity_continuation_local_v0'),
                     ('conductivity_local_read_decision','conductivity_read_decision_local_v0'),
                     ('conductivity_glm_read_decision','conductivity_read_decision_glm_v0'),
                     ('conductivity_hybrid_read_decision','conductivity_read_decision_hybrid_v0'),
                     ('conductivity_task_memory_swaps','conductivity_read_decision_swaps_v0'),
                     ('conductivity_exact_actor_prompts','conductivity_actor_prompt_repeats_v0'),
                     ('conductive_counterpart_curriculum','conductive_counterpart_local_v1'),
                     ('conductive_counterpart_local_actor','conductive_counterpart_actor_local_v0'),
                     ('conductive_counterpart_hybrid_actor','conductive_counterpart_actor_hybrid_v0'),
                     ('conductivity_complete_learning','conductivity_complete_learning_local_v0'),
                     ('conductivity_complete_hybrid_actor','conductivity_complete_learning_actor_hybrid_v0'),
                     ('conductivity_complete_glm52_actor','conductivity_complete_actor_glm52_v0'),
                     ('conductivity_corrective_users','conductivity_recovery_local_v0'),
                     ('conductivity_calibrated_gate','conductivity_calibrated_matched_local_v0'),
                     ('conductivity_shared_users','conductivity_shared_users_hybrid_v0'),
                     ('conductivity_official_flash_actor','conductivity_official_deepseek_actor_v0'),
                     ('conductivity_official_alias_calibration','conductivity_calibrated_alias_official_v0'),
                     ('conductivity_official_pro_calibration','conductivity_calibrated_official_pro_v0'),
                     ('worldmind_native_process_replay','worldmind_conductivity_replay_local_v0'),
                     ('worldmind_native_retrieval_actor','worldmind_retrieval_official_actor_v1'),
                     ('official_pro_rule_gate','erasure_official_pro_gate_v0'),
                     ('native_cross_task_erasure','native_cross_task_erasure_pro_v0'),
                     ('cross_task_erasure_actor_flash','cross_task_erasure_actor_flash_v0'),
                     ('cross_task_erasure_actor_pro','cross_task_erasure_actor_pro_v0'),
                     ('cross_task_erasure_actor_repeats','cross_task_erasure_flash_repeats_v0'),
                     ('cross_task_erasure_no_debit','native_cross_task_erasure_pro_no_debit_v0'),
                     ('cross_task_erasure_no_debit_matched','native_cross_task_erasure_pro_no_debit_matched_v0'),
                     ('tdmpc_clean_gate_initial','tdmpc_clean_gate_v0'),
                     ('tdmpc_clean_gate_followup','tdmpc_clean_gate_v1'),
                     ('tdmpc_real_data_adaptation','tdmpc_adaptation_screen_v0'),
                     ('tdmpc_shared_physics_adaptation','tdmpc_adaptation_screen_v1'),
                     ('tdmpc_native_full_adaptation','tdmpc_native_adaptation_v0'),
                     ('tdmpc_targeted_action_search','tdmpc_action_search_v0'),
                     ('tdmpc_actual_return_search','tdmpc_return_search_v0'),
                     ('tdmpc_update_dose','tdmpc_update_dose_v0'),
                     ('pro_forecast_assisted_user_chain','pro_erasure_shared_users_v0'),
                     ('pro_rule_relearning_replay','pro_rule_relearning_interventions_v0'),
                     ('pro_rule_relearning_queries','pro_rule_relearning_queries_v0'),
                     ('compositional_plan_execution','compositional_plans_pro_v0'),
                     ('thermal_curriculum','thermal_curriculum_local_v2'),
                     ('thermal_local_actor','thermal_actor_local_v0'),
                     ('thermal_hybrid_actor','thermal_actor_hybrid_glm52_v0')]:
    p=ROOT/'runs'/folder/'status.json'
    if p.exists():diagnostics[label]={'source':str(p.relative_to(ROOT)),'result':json.loads(p.read_text())}

for label,folder in [('glm_capture_transfer','capture_reads_glm'),
                     ('local_recency_intervention','capture_reads_local_recency')]:
    p=ROOT/'runs'/folder/'results.json'
    if p.exists():
        diagnostics[label]={'source':str(p.relative_to(ROOT)),
            'result':[{'condition':r['condition'],'view':r['view'],'repeat':r['repeat'],
                       'coded_move':move_outcome(r['prediction']['prediction']),
                       'retrieved_indices':r['prediction']['retrieved_indices']}
                      for r in json.loads(p.read_text())]}

for folder in ['conductivity_glm_v0','conductivity_reads_local','conductivity_history_local_v0',
               'conductive_counterpart_local_v1','conductivity_complete_learning_local_v0',
               'conductivity_complete_reads_glm52_v0','conductivity_official_deepseek_reads_v0',
               'conductivity_calibrated_official_pro_v0','conductivity_pro_rule_protection_v0',
               'conductivity_pro_after_success_queries_v0','pro_rule_relearning_queries_v0',
               'thermal_curriculum_local_v2']:
    p=ROOT/'runs'/folder
    files=[p/'results.json'] if (p/'results.json').exists() else list(p.glob('*_result.json'))
    result=[]
    for f in files:
        if not f.exists():continue
        d=json.loads(f.read_text())
        items=d if isinstance(d,list) else d.get('evaluation',[])
        for r in items:
            result.append(dict(condition=r.get('condition',d.get('arm') if isinstance(d,dict) else None),
                view=r.get('query',r.get('view')),prediction=r.get('prediction',{}).get('prediction'),
                actual=r.get('actual'),source=str(f.relative_to(ROOT))))
    diagnostics[folder]={'result':result}

for folder in ['native_cross_task_erasure_pro_v0', 'native_cross_task_erasure_pro_no_debit_matched_v0']:
    p=ROOT/'runs'/folder/'results.json'
    if p.exists():
        diagnostics[folder+'_forecasts']={'source':str(p.relative_to(ROOT)), 'result':[
            {'condition':r['condition'], 'executed_actions':r['executed_actions'], 'evidence':r['evidence'],
             'forecasts':[{'view':q['view'], 'lamp_on':q['lamp_on'],
                          'prediction':q['prediction']['prediction']} for q in r['evaluation']]}
            for r in json.loads(p.read_text())]}

p=ROOT/'runs/pro_erasure_shared_users_v0/probes.json'
if p.exists():
    diagnostics['pro_forecast_assisted_user_chain_probes']={
        'source':str(p.relative_to(ROOT)), 'result':[
            dict(condition=r['condition'], completed_users=r['completed_users'],
                 view=r['view'], prediction=r['prediction']['prediction'],
                 actual=r['actual']) for r in json.loads(p.read_text())]}
for label, filename in [
    ('tdmpc_action_search_returns','runs/tdmpc_action_search_v0/results.json'),
    ('tdmpc_return_search_selection','runs/tdmpc_return_search_v0/selection.json'),
    ('tdmpc_update_dose_returns','runs/tdmpc_update_dose_v0/results.json'),
    ('pro_observed_rule_reextractions','runs/pro_rule_relearning_interventions_v0/merge_replay.json')]:
    p=ROOT/filename
    if p.exists():diagnostics[label]={'source':filename,'result':json.loads(p.read_text())}

p=ROOT/'runs/compositional_plans_pro_v0/results.json'
if p.exists():
    diagnostics['compositional_plan_execution_outcomes']={'source':str(p.relative_to(ROOT)), 'result':[
        dict(task=r['task'],status=r['status'],attack_specific_failure=r.get('attack_specific_failure'),
             evaluations={k:{field:v[field] for field in ['selected','task_success','all_feasibility_correct']}
                          for k,v in r.get('evaluations',{'clean':r.get('clean')}).items() if v})
        for r in json.loads(p.read_text())]}

from summarize_discoveryworld_gate import collect as collect_discoveryworld, RUNS as DISCOVERYWORLD_RUNS
discoveryworld = {key:collect_discoveryworld(folder) for key,folder in DISCOVERYWORLD_RUNS.items()}
discoveryworld['plant_flash_high_thinking'] = collect_discoveryworld('discoveryworld_plant_flash_reasoning_clean_v0')
discoveryworld['plant_flash_high_v1'] = collect_discoveryworld('discoveryworld_plant_flash_reasoning_clean_v1')
discoveryworld['plant_gateway_v1'] = collect_discoveryworld('discoveryworld_plant_glm53_clean_v1')
diagnostics['discoveryworld_native_capability'] = {'result':discoveryworld,
    'scope':'Development capability and shared-evidence benefit gate. Source-curriculum attack attempts are reported separately; no downstream attack effect established.'}
reference_config=json.loads((ROOT/'configs/discoveryworld_reference_screen.json').read_text())
diagnostics['discoveryworld_reference_source_screen']={
    'conditions':{key:collect_discoveryworld(Path(folder).name) for key,folder in reference_config.get('source_runs',{}).items()},
    'scope':reference_config['scope'], 'hypothesis':reference_config['hypothesis'],
    'result':reference_config.get('result'), 'decision':reference_config.get('decision')}
success_calibration_config=json.loads((ROOT/'configs/discoveryworld_success_calibration_screen.json').read_text())
diagnostics['discoveryworld_success_calibration_source_screen']={
    'conditions':{key:collect_discoveryworld(Path(folder).name) for key,folder in success_calibration_config.get('source_runs',{}).items()},
    'scope':success_calibration_config['scope'], 'hypothesis':success_calibration_config['hypothesis'],
    'result':success_calibration_config.get('result'), 'decision':success_calibration_config.get('decision')}
plant_relocation_config=json.loads((ROOT/'configs/discoveryworld_plant_relocation_screen.json').read_text())
diagnostics['discoveryworld_plant_relocation_source_screen']={
    'conditions':{key:collect_discoveryworld(Path(folder).name) for key,folder in plant_relocation_config.get('source_runs',{}).items()},
    'scope':plant_relocation_config['novelty_limit'], 'hypothesis':plant_relocation_config['hypothesis'],
    'status':plant_relocation_config['status'],
    'result':plant_relocation_config.get('result',plant_relocation_config.get('partial_result')),
    'decision':plant_relocation_config.get('decision'),
    'victim_result':plant_relocation_config.get('victim_result'),
    'victims':{key:collect_discoveryworld(Path(folder).name)
               for key,folder in plant_relocation_config.get('victim_runs',{}).items()}}

causalgame_config=json.loads((ROOT/'configs/causalgame_feasibility.json').read_text())
causalgame_run=causalgame_config.get('capability_run',{})
causalgame_result_path=ROOT/causalgame_run.get('output','runs/causalgame_unrun')/'summary.json'
diagnostics['causalgame_native_capability_gate']={
    'status':causalgame_config['status'],
    'plan':causalgame_run,
    'result':json.loads(causalgame_result_path.read_text()) if causalgame_result_path.exists() else None,
    'excluded_runs':causalgame_config.get('excluded_runs',[]),
    'novelty_limit':causalgame_config['novelty_gate']}

guardrail_review=json.loads((ROOT/'configs/predictive_guardrail_candidate.json').read_text())
diagnostics['predictive_guardrail_candidate_review']={
    'source':'configs/predictive_guardrail_candidate.json','result':guardrail_review}
query_interference=json.loads((ROOT/'configs/tdmpc_query_interference_screen.json').read_text())
diagnostics['tdmpc_query_only_planner_interference']={
    'source':'configs/tdmpc_query_interference_screen.json','result':query_interference}
api_endpoints=json.loads((ROOT/'configs/api_endpoints_status.json').read_text())
diagnostics['api_endpoints_status']={
    'source':'configs/api_endpoints_status.json','result':api_endpoints}
drwell_credit=json.loads((ROOT/'configs/drwell_success_credit_screen.json').read_text())
diagnostics['drwell_success_credit_screen']={
    'source':'configs/drwell_success_credit_screen.json','result':drwell_credit}
baba_rejection=json.loads((ROOT/'configs/baba_rejection_learning_screen.json').read_text())
diagnostics['baba_rejection_learning_screen']={
    'source':'configs/baba_rejection_learning_screen.json','result':baba_rejection}
experiment_transport=json.loads((ROOT/'configs/experiment_transport_screen.json').read_text())
diagnostics['experiment_transport_screen']={
    'source':'configs/experiment_transport_screen.json','result':experiment_transport}
interphyre_program=json.loads((ROOT/'configs/interphyre_program_search_screen.json').read_text())
diagnostics['interphyre_program_search_screen']={
    'source':'configs/interphyre_program_search_screen.json','result':interphyre_program}
interphyre_continuous=json.loads((ROOT/'configs/interphyre_continuous_search_screen.json').read_text())
diagnostics['interphyre_continuous_search_screen']={
    'source':'configs/interphyre_continuous_search_screen.json','result':interphyre_continuous}
interphyre_generated=json.loads((ROOT/'configs/interphyre_policy_generation_screen.json').read_text())
diagnostics['interphyre_policy_generation_screen']={
    'source':'configs/interphyre_policy_generation_screen.json','result':interphyre_generated}
neuronbench_screen=json.loads((ROOT/'configs/neuronbench_mechanism_screen.json').read_text())
diagnostics['neuronbench_mechanism_screen']={
    'source':'configs/neuronbench_mechanism_screen.json','result':neuronbench_screen}
sampling_security=json.loads((ROOT/'configs/world_model_sampling_security.json').read_text())
diagnostics['world_model_sampling_security']={
    'source':'configs/world_model_sampling_security.json','result':sampling_security}
gradient_security=json.loads((ROOT/'configs/world_model_gradient_security.json').read_text())
diagnostics['world_model_gradient_security']={
    'source':'configs/world_model_gradient_security.json','result':gradient_security}
origin_security=json.loads((ROOT/'configs/world_model_origin_security.json').read_text())
diagnostics['world_model_origin_security']={
    'source':'configs/world_model_origin_security.json','result':origin_security}
recurrence_security=json.loads((ROOT/'configs/world_model_recurrence_security.json').read_text())
diagnostics['world_model_recurrence_security']={
    'source':'configs/world_model_recurrence_security.json','result':recurrence_security}
geometry_security=json.loads((ROOT/'configs/world_model_geometry_security.json').read_text())
diagnostics['world_model_geometry_security']={
    'source':'configs/world_model_geometry_security.json','result':geometry_security}
goal_isolation=json.loads((ROOT/'configs/world_model_goal_isolation_screen.json').read_text())
diagnostics['world_model_goal_isolation']={
    'source':'configs/world_model_goal_isolation_screen.json','result':goal_isolation}
goal_bus=json.loads((ROOT/'configs/world_model_goal_bus_security.json').read_text())
diagnostics['world_model_goal_bus_security']={
    'source':'configs/world_model_goal_bus_security.json','result':goal_bus}
uncertainty_screen=json.loads((ROOT/'configs/world_model_uncertainty_screen.json').read_text())
diagnostics['world_model_uncertainty_decomposition']={
    'source':'configs/world_model_uncertainty_screen.json','result':uncertainty_screen}
rejected_future=json.loads((ROOT/'configs/world_model_rejected_future_screen.json').read_text())
diagnostics['world_model_rejected_future']={
    'source':'configs/world_model_rejected_future_screen.json','result':rejected_future}
dynamics_reliance=json.loads((ROOT/'configs/tdmpc_dynamics_reliance_screen.json').read_text())
diagnostics['tdmpc_dynamics_reliance']={
    'source':'configs/tdmpc_dynamics_reliance_screen.json','result':dynamics_reliance}
policy_symmetry=json.loads((ROOT/'configs/world_model_policy_symmetry_screen.json').read_text())
diagnostics['world_model_policy_symmetry']={
    'source':'configs/world_model_policy_symmetry_screen.json','result':policy_symmetry}
private_counterfactual_path=ROOT/'configs/world_model_private_counterfactual_screen.json'
if private_counterfactual_path.exists():
    private_counterfactual=json.loads(private_counterfactual_path.read_text())
    diagnostics['world_model_private_counterfactual']={
        'source':'configs/world_model_private_counterfactual_screen.json',
        'result':private_counterfactual}
temporal_correlation_path=ROOT/'configs/world_model_temporal_correlation_screen.json'
if temporal_correlation_path.exists():
    temporal_correlation=json.loads(temporal_correlation_path.read_text())
    diagnostics['world_model_temporal_correlation']={
        'source':'configs/world_model_temporal_correlation_screen.json',
        'result':temporal_correlation}
epistemic_collapse_path=ROOT/'configs/world_model_epistemic_collapse_screen.json'
if epistemic_collapse_path.exists():
    epistemic_collapse=json.loads(epistemic_collapse_path.read_text())
    diagnostics['world_model_epistemic_collapse']={
        'source':'configs/world_model_epistemic_collapse_screen.json',
        'result':epistemic_collapse}
conductivity_gemini_path=ROOT/'configs/conductivity_gemini_actor_screen.json'
if conductivity_gemini_path.exists():
    conductivity_gemini=json.loads(conductivity_gemini_path.read_text())
    diagnostics['conductivity_gemini_actor']={
        'source':'configs/conductivity_gemini_actor_screen.json',
        'result':conductivity_gemini}
trajectory_order_path=ROOT/'configs/world_model_trajectory_order_screen.json'
if trajectory_order_path.exists():
    trajectory_order=json.loads(trajectory_order_path.read_text())
    diagnostics['world_model_trajectory_order']={
        'source':'configs/world_model_trajectory_order_screen.json',
        'result':trajectory_order}
action_contrast_path=ROOT/'configs/world_model_action_contrast_screen.json'
if action_contrast_path.exists():
    action_contrast=json.loads(action_contrast_path.read_text())
    diagnostics['world_model_action_contrast']={
        'source':'configs/world_model_action_contrast_screen.json',
        'result':action_contrast}
terminal_ownership_path=ROOT/'configs/world_model_terminal_ownership_screen.json'
if terminal_ownership_path.exists():
    terminal_ownership=json.loads(terminal_ownership_path.read_text())
    diagnostics['world_model_terminal_ownership']={
        'source':'configs/world_model_terminal_ownership_screen.json',
        'result':terminal_ownership}
branch_provenance_path=ROOT/'configs/world_model_branch_provenance_screen.json'
if branch_provenance_path.exists():
    branch_provenance=json.loads(branch_provenance_path.read_text())
    diagnostics['world_model_branch_provenance']={
        'source':'configs/world_model_branch_provenance_screen.json',
        'result':branch_provenance}
padded_action_path=ROOT/'configs/world_model_padded_action_screen.json'
if padded_action_path.exists():
    padded_action=json.loads(padded_action_path.read_text())
    diagnostics['world_model_padded_action']={
        'source':'configs/world_model_padded_action_screen.json',
        'result':padded_action}
optimizer_state_path=ROOT/'configs/world_model_optimizer_state_screen.json'
if optimizer_state_path.exists():
    optimizer_state=json.loads(optimizer_state_path.read_text())
    diagnostics['world_model_optimizer_state']={
        'source':'configs/world_model_optimizer_state_screen.json',
        'result':optimizer_state}
temporal_gradient_dither_path=ROOT/'configs/world_model_temporal_gradient_dither_screen.json'
if temporal_gradient_dither_path.exists():
    temporal_gradient_dither=json.loads(temporal_gradient_dither_path.read_text())
    diagnostics['world_model_temporal_gradient_dither']={
        'source':'configs/world_model_temporal_gradient_dither_screen.json',
        'result':temporal_gradient_dither}
gemini_search_path=ROOT/'runs/gemini_mechanism_search_20261009T1155Z/summary.json'
if gemini_search_path.exists():
    gemini_search=json.loads(gemini_search_path.read_text())
    diagnostics['gemini_targeted_mechanism_search']={
        'source':'runs/gemini_mechanism_search_20261009T1155Z/summary.json',
        'result':gemini_search}
gemini_hidden_interface_path=ROOT/'runs/gemini_hidden_interface_search_20261009T1320Z/summary.json'
if gemini_hidden_interface_path.exists():
    gemini_hidden_interface=json.loads(gemini_hidden_interface_path.read_text())
    diagnostics['gemini_hidden_interface_search']={
        'source':'runs/gemini_hidden_interface_search_20261009T1320Z/summary.json',
        'result':gemini_hidden_interface}
gemini_post_screen_path=ROOT/'configs/gemini_post_novelty_screen.json'
if gemini_post_screen_path.exists():
    gemini_post_screen=json.loads(gemini_post_screen_path.read_text())
    diagnostics['gemini_post_novelty_screen']={
        'source':'configs/gemini_post_novelty_screen.json',
        'result':gemini_post_screen}
gemini_strict_new_path=ROOT/'runs/gemini_strict_new_candidate_20261009T0000Z/summary.json'
if gemini_strict_new_path.exists():
    gemini_strict_new=json.loads(gemini_strict_new_path.read_text())
    diagnostics['gemini_strict_new_candidate']={
        'source':'runs/gemini_strict_new_candidate_20261009T0000Z/summary.json',
        'result':gemini_strict_new}
wav_alias_path=ROOT/'configs/world_model_wav_alias_screen.json'
if wav_alias_path.exists():
    wav_alias=json.loads(wav_alias_path.read_text())
    diagnostics['world_model_wav_action_alias']={
        'source':'configs/world_model_wav_alias_screen.json',
        'result':wav_alias}
boundary_stiffness_path=ROOT/'configs/world_model_boundary_stiffness_screen.json'
if boundary_stiffness_path.exists():
    boundary_stiffness=json.loads(boundary_stiffness_path.read_text())
    diagnostics['world_model_boundary_stiffness']={
        'source':'configs/world_model_boundary_stiffness_screen.json',
        'result':boundary_stiffness}
replay_priority_path=ROOT/'configs/world_model_replay_priority_screen.json'
if replay_priority_path.exists():
    replay_priority=json.loads(replay_priority_path.read_text())
    diagnostics['world_model_replay_priority']={
        'source':'configs/world_model_replay_priority_screen.json',
        'result':replay_priority}
timeout_laundering_path=ROOT/'configs/world_model_timeout_laundering_screen.json'
if timeout_laundering_path.exists():
    timeout_laundering=json.loads(timeout_laundering_path.read_text())
    diagnostics['world_model_timeout_laundering']={
        'source':'configs/world_model_timeout_laundering_screen.json',
        'result':timeout_laundering}
non_normal_path=ROOT/'configs/world_model_non_normal_screen.json'
if non_normal_path.exists():
    non_normal=json.loads(non_normal_path.read_text())
    diagnostics['world_model_non_normal_transient']={
        'source':'configs/world_model_non_normal_screen.json',
        'result':non_normal}
contractive_basin_path=ROOT/'configs/world_model_contractive_basin_screen.json'
if contractive_basin_path.exists():
    contractive_basin=json.loads(contractive_basin_path.read_text())
    diagnostics['world_model_contractive_basin']={
        'source':'configs/world_model_contractive_basin_screen.json',
        'result':contractive_basin}
horizon_shadowing_path=ROOT/'configs/world_model_horizon_shadowing_screen.json'
if horizon_shadowing_path.exists():
    horizon_shadowing=json.loads(horizon_shadowing_path.read_text())
    diagnostics['world_model_horizon_shadowing']={
        'source':'configs/world_model_horizon_shadowing_screen.json',
        'result':horizon_shadowing}
non_normal_causal_path=ROOT/'runs/tdmpc_non_normal_causal_h1_dose300_v0/status.json'
if non_normal_causal_path.exists():
    non_normal_causal=json.loads(non_normal_causal_path.read_text())
    diagnostics['world_model_non_normal_causal_h1']={
        'source':'runs/tdmpc_non_normal_causal_h1_dose300_v0/status.json',
        'result':non_normal_causal}
non_normal_clean_h1_path=ROOT/'runs/tdmpc_non_normal_clean_horizon1_v0/status.json'
if non_normal_clean_h1_path.exists():
    non_normal_clean_h1=json.loads(non_normal_clean_h1_path.read_text())
    diagnostics['world_model_non_normal_clean_h1']={
        'source':'runs/tdmpc_non_normal_clean_horizon1_v0/status.json',
        'result':non_normal_clean_h1}
non_normal_selective_path=ROOT/'runs/tdmpc_non_normal_selective_horizon_fulljac_dose300_v0/status.json'
if non_normal_selective_path.exists():
    non_normal_selective=json.loads(non_normal_selective_path.read_text())
    diagnostics['world_model_non_normal_selective_horizon']={
        'source':'runs/tdmpc_non_normal_selective_horizon_fulljac_dose300_v0/status.json',
        'result':non_normal_selective}
parity_screen_path=ROOT/'configs/world_model_parity_coupling_screen.json'
if parity_screen_path.exists():
    parity_screen=json.loads(parity_screen_path.read_text())
    diagnostics['world_model_parity_coupling_screen']={
        'source':'configs/world_model_parity_coupling_screen.json',
        'result':parity_screen}
composition_screen_path=ROOT/'configs/world_model_composition_integrity_screen.json'
if composition_screen_path.exists():
    composition_screen=json.loads(composition_screen_path.read_text())
    diagnostics['world_model_composition_integrity_screen']={
        'source':'configs/world_model_composition_integrity_screen.json',
        'result':composition_screen}
frequency_semigroup_path=ROOT/'configs/world_model_frequency_semigroup_screen.json'
if frequency_semigroup_path.exists():
    frequency_semigroup=json.loads(frequency_semigroup_path.read_text())
    diagnostics['world_model_frequency_semigroup_screen']={
        'source':'configs/world_model_frequency_semigroup_screen.json',
        'result':frequency_semigroup}
parity_exact_path=ROOT/'runs/parity_coupling_exact_v0/summary.json'
if parity_exact_path.exists():
    diagnostics['world_model_parity_exact']={
        'source':'runs/parity_coupling_exact_v0/summary.json',
        'result':json.loads(parity_exact_path.read_text())}
commutator_path=ROOT/'runs/tdmpc_commutator_diagnostic_v0/status.json'
if commutator_path.exists():
    diagnostics['world_model_commutator_diagnostic']={
        'source':'runs/tdmpc_commutator_diagnostic_v0/status.json',
        'result':json.loads(commutator_path.read_text())}
collective_review=json.loads((ROOT/'configs/collective_learning_candidate_screen.json').read_text())
diagnostics['collective_learning_candidate_screen']={
    'source':'configs/collective_learning_candidate_screen.json','result':collective_review}
scale_carryover=json.loads((ROOT/'configs/tdmpc_scale_carryover_screen.json').read_text())
diagnostics['tdmpc_scale_carryover_screen']={
    'source':'configs/tdmpc_scale_carryover_screen.json','result':scale_carryover}
mechanism_review=json.loads((ROOT/'configs/mechanism_review_20261008.json').read_text())
diagnostics['mechanism_review_20261008']={
    'source':'configs/mechanism_review_20261008.json','result':mechanism_review}
dmo_jacobian=json.loads((ROOT/'configs/dmo_jacobian_screen.json').read_text())
diagnostics['dmo_jacobian_screen']={
    'source':'configs/dmo_jacobian_screen.json','result':dmo_jacobian}
repair_transport=json.loads((ROOT/'configs/repair_transport_screen.json').read_text())
diagnostics['repair_transport_screen']={
    'source':'configs/repair_transport_screen.json','result':repair_transport}
opine_memory=json.loads((ROOT/'configs/opine_query_memory_screen.json').read_text())
diagnostics['opine_query_memory_screen']={
    'source':'configs/opine_query_memory_screen.json','result':opine_memory}
opine_sync=json.loads((ROOT/'configs/opine_state_synchronization_screen.json').read_text())
diagnostics['opine_state_synchronization_screen']={
    'source':'configs/opine_state_synchronization_screen.json','result':opine_sync}
reactor_grammar=json.loads((ROOT/'configs/reactor_program_grammar_screen.json').read_text())
diagnostics['reactor_program_grammar_screen']={
    'source':'configs/reactor_program_grammar_screen.json','result':reactor_grammar}
acquisition_search=json.loads((ROOT/'configs/reactor_acquisition_search_screen.json').read_text())
diagnostics['reactor_acquisition_search_screen']={
    'source':'configs/reactor_acquisition_search_screen.json','result':acquisition_search}
lazy_calibration=json.loads((ROOT/'configs/reactor_lazy_calibration_screen.json').read_text())
diagnostics['reactor_lazy_calibration_screen']={
    'source':'configs/reactor_lazy_calibration_screen.json','result':lazy_calibration}
evidence_revision=json.loads((ROOT/'configs/reactor_evidence_revision_screen.json').read_text())
diagnostics['reactor_evidence_revision_screen']={
    'source':'configs/reactor_evidence_revision_screen.json','result':evidence_revision}

usage=defaultdict(lambda:{'requests':0,'failed_requests':0,'input_tokens':0,'output_tokens':0,'latency_s':0.0})
for p in (ROOT/'runs').rglob('calls.jsonl'):
    for line in p.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        k=r['request']['model'];u=usage[k];u['requests']+=1
        u['failed_requests']+=int(bool(r.get('error')))
        t=r.get('response',{}).get('usage',{})
        u['input_tokens']+=t.get('prompt_tokens',0);u['output_tokens']+=t.get('completion_tokens',0)
        u['latency_s']+=r.get('latency_s',0)

# A researcher interrupt can occur after the transport increments its counter
# but before it writes a response/error record. Count the known started attempt
# without inventing response tokens, latency, or an endpoint failure.
for p in (ROOT/'runs').glob('causalgame_native*/summary.json'):
    d=json.loads(p.read_text())
    plan_path=p.parent/'plan.json'
    calls_path=p.parent/'calls.jsonl'
    if not plan_path.exists() or not d.get('api_requests'):
        continue
    recorded=len(calls_path.read_text().splitlines()) if calls_path.exists() else 0
    unrecorded=max(0,d['api_requests']-recorded)
    if unrecorded:
        model=json.loads(plan_path.read_text())['model']
        usage[model]['requests']+=unrecorded
        usage[model]['interrupted_without_response_record']=usage[model].get('interrupted_without_response_record',0)+unrecorded

summary={'stage':'development; main claim unproven; not a conference-scale result',
         'completed_results':rows,'usage_including_failed_requests':dict(usage),
         'development_diagnostics':diagnostics,'native_scienceworld_runs':native_runs,
         'interpretation':'Stable task loss remains unestablished. One historical Qwen3-WM/glm-5.3-actor development continuation scored100/100/-100; subsequent controlled studies do not establish a reliable effect. Pro-WM/Flash-actor six fixed repetitions per condition all succeed, and all nine continuations in three persistent-user chains also succeed. TD-MPC2 action-only search changes a model proxy but not held-out task performance; actual-return selection misses the preset100-point calibration gate, and120/300-update pilots show only small declines. No large-scale training warranted.',
         'repair_evidence':'Each of six independently repeated successful corrections leaves both Pro bulb queries wrong. Three sequential forecast-assisted users also leave attacked predictions wrong, but the ordinary-learning control degrades by user3 despite positive bulb-rule evidence. All tasks succeed. Three observed native extractions reproduce exactly the dormant correct look-at rule; upstream duplicate merging refreshes its timestamp without evidence, keeping it hidden. This is an implementation-specific repair failure, not irreversible knowledge loss or an attack-specific long-term task effect.',
         'next_mechanism':'Official Pro resists calibrated bypass learning. Subsequent unrelated genuine thermal measurements cause 28/30 prediction mismatches and debit bulb rules; the circuit forecast becomes wrong by step20, while the inspect-thermometer control remains correct. Disabling global negative credit from the identical post-setup checkpoint preserves both query views through30 steps. Disabling it earlier does not prevent setup degradation, so the debit is a scoped mechanism, not a complete explanation or general defense. Task harm is not established: six fixed full repetitions per clean/benign/attack condition with Pro WM and Flash actor all score100 (18/18); all attacked actors measure and correct. All-Pro actor fails its clean completion gate. This case is prediction harm only and does not meet the scale-up gate. SWAAP already studies latent-world-model data poisoning and cross-task transfer; switching to MPC alone supplies no novelty.'}
summary['current_research']={
    'claim_status':'No accepted final claim. The epistemic-disagreement collapse screen shows a local task-loss signal but fails its preregistered two-seed promotion gate and has an uncertainty-attack novelty confound.',
    'accepted_claim':None,
    'latest_result_cn':(epistemic_collapse['report_note_cn'] if epistemic_collapse_path.exists() else (temporal_correlation['report_note_cn'] if temporal_correlation_path.exists() else (private_counterfactual['report_note_cn'] if private_counterfactual_path.exists() else policy_symmetry['report_note_cn']))),
    'source':('configs/world_model_epistemic_collapse_screen.json' if epistemic_collapse_path.exists() else ('configs/world_model_temporal_correlation_screen.json' if temporal_correlation_path.exists() else ('configs/world_model_private_counterfactual_screen.json' if private_counterfactual_path.exists() else 'configs/world_model_policy_symmetry_screen.json'))),
    'native_task_baselines':'configs/tdmpc_dynamics_reliance_screen.json',
    'rejected_future_screen':'configs/world_model_rejected_future_screen.json',
    'uncertainty_screen':'configs/world_model_uncertainty_screen.json',
    'goal_isolation_screen':'configs/world_model_goal_isolation_screen.json',
    'coordinate_screen':'configs/world_model_geometry_security.json',
    'recurrence_screen':'configs/world_model_recurrence_security.json',
    'limited_rank_evidence':'configs/world_model_sampling_security.json',
    'not_promoted_gradient_construction':'configs/world_model_gradient_security.json',
    'origin_discrimination_screen':'configs/world_model_origin_security.json',
    'private_counterfactual_screen':'configs/world_model_private_counterfactual_screen.json',
    'temporal_correlation_screen':'configs/world_model_temporal_correlation_screen.json',
    'epistemic_collapse_screen':'configs/world_model_epistemic_collapse_screen.json',
    'conductivity_gemini_actor_screen':'configs/conductivity_gemini_actor_screen.json',
    'gemini_post_novelty_screen':'configs/gemini_post_novelty_screen.json',
    'trajectory_order_screen':'configs/world_model_trajectory_order_screen.json',
    'action_contrast_screen':'configs/world_model_action_contrast_screen.json',
    'gemini_targeted_mechanism_search':'runs/gemini_mechanism_search_20261009T1155Z/summary.json',
    'next_requirement':(epistemic_collapse['next_requirement'] if epistemic_collapse_path.exists() else (temporal_correlation['next_requirement'] if temporal_correlation_path.exists() else (private_counterfactual['next_requirement'] if private_counterfactual_path.exists() else policy_symmetry['next_requirement'])))}
summary['historical_interpretation']=summary['interpretation']
summary['interpretation']='No accepted final security claim. Strong-baseline follow-up limits rank coupling: upstream-default CartPole CEM scores 200 in both conditions; pretrained TD-MPC2 scores 974.27 versus 973.40. Earlier reduced-CEM and exact-model positive examples remain limited evidence. The subsequent prediction-law-preserving sampler-gradient construction is not promoted as a novel contribution: its source-code permissions and known gradient-estimator mechanism leave a novelty gap, while its frozen Gaussian-model gradient MPC clean gate scores only 364.32/1000 against a preset 900 threshold. No gradient attack or repair condition was run. Stop scaling these fixed constructions; seek a substantively distinct security operation with a competent applicable controller.'
summary['interpretation']+=' The latest origin-discrimination screen uses an unchanged strong native TD-MPC2 controller (994.12 and 974.09 on two new episodes). The frozen classifier has strong imagined-state detection but real false-positive rates0.80% and1.61%; the latter fails the preset1% limit. A structural LayerNorm necessary-condition probe does not separate real and imagined states. No world-model attack or task harm was tested; origin classification alone is not novel OOD routing.'
summary['interpretation']+=' A subsequent dynamics-only rank-one weight edit (120 updates,6.16 seconds) preserves one-step fidelity but misses a preset paired-action decision gate. Replaying clean-path logit perturbations at every step without edit feedback reproduces or exceeds the effect; this does not support the self-excitation hypothesis. No native task attack was run. The user-selected official Gemini API passed both a short JSON request and a research-length request; these are integration checks, not research-claim evidence.'
summary['historical_next_mechanism']=summary['next_mechanism']
summary['interpretation']+=' A subsequent CPU algebra screen confirms that a conjugated latent coordinate system changes decisions only through the untransported cost. Linear whitening restores equivalence; a constructed Gaussian-preserving nonlinear twist evades whitening but changes nonzero latent prediction error. This overlaps existing latent-geometry work and supplies no accepted new operation or native task-harm evidence. A third official Gemini generation succeeded; its critique was independently checked.'
summary['interpretation']+=' The WorldAgen goal-bus replication now covers 30 recorded LIBERO windows across 10 episodes from four public LIBERO-10 parquet files. With images, states, and six actions fixed, swapping the goal changes world prediction by relative L2 0.300 on average (empty goal 0.270); cutting the nine native goal-to-input-action edges makes every world output exactly invariant while policy output remains goal-sensitive (relative L2 0.176). The recorded-goal, swapped-goal, and edge-cut primary forecast MSE are 1.423, 1.496, and 1.484, exposing a measurable accuracy tradeoff. A prior nine-window replication remains archived. This promotes the shared action-token causal bus to the main candidate, while cross-backbone and closed-loop task replication remain required. The earlier three-clip CPU result and exact checkpoint loading remain preserved.'
summary['interpretation']+=' A further one-forward CPU screen on 64 fixed recorded transitions rejects a proposed native Gaussian uncertainty redistribution: means and diagonal variances match, but coordinate CDFs differ by up to 0.01910. An exact nested-mixture identity requires a different output family, and resampling persistent model identities can change trajectory laws. No planner attack or training was run. Exact predictive-law preservation is not imposed as a requirement on future research ideas.'
summary['interpretation']+=' The subsequent rejected-future screen stops before any forecast edit: the composed MiniGrid selector reaches only 62/128 fixed goals, below the preset80% clean gate; goal-only reaches56/128. All128 goals match recorded next states. This is not a native WAV or ACID reproduction, and normalization-mediated dominated-decoy effects already have direct prior art. No attack, training, or GPU experiment was run.'
summary['interpretation']+=' A subsequent causal native-control baseline runs nine full500-step episodes across walker-walk, reacher-easy and cup-catch. Replacing the dynamics action input with zeros gives returns80.22/946/718 versus native975.51/972/994; native policy-only returns500.25/973/996. This identifies useful contrasting tasks without changing any learned parameter. It is a one-seed ablation, not a new security mechanism or a stealthy attack. No training was run.'
summary['next_mechanism']=policy_symmetry['next_requirement']
summary['interpretation']+=' '+policy_symmetry['report_note_cn']
if private_counterfactual_path.exists():
    summary['interpretation']+=' '+private_counterfactual['report_note_cn']
if temporal_correlation_path.exists():
    summary['interpretation']+=' '+temporal_correlation['report_note_cn']
if epistemic_collapse_path.exists():
    summary['interpretation']+=' '+epistemic_collapse['report_note_cn']
if conductivity_gemini_path.exists():
    summary['interpretation']+=' '+conductivity_gemini['report_note_cn']
if gemini_post_screen_path.exists():
    summary['interpretation']+=' '+gemini_post_screen['finding_cn']+' '+gemini_post_screen['decision_cn']
if gemini_strict_new_path.exists():
    summary['interpretation']+=' 最新严格 Gemini 筛选在汇总既有负面短测和文献约束后没有返回可接受候选；该请求仅用于机制筛选，不构成实验结果，也没有启动 GPU 扩量。'
if wav_alias_path.exists():
    summary['interpretation']+=' WAV 的动作别名诊断在四个已发布检查点上把151组物理端点等价动作中的141—146组拆成不同模型后继；严格 Gemini 评审将其判为普通 OOD 动作语义/等价性泛化误差（新意2/10），没有攻击操作或原生控制器任务损害，因此仅保留为实现诊断，不升级为安全主张。'
if boundary_stiffness_path.exists():
    summary['interpretation']+=' 合法硬边界振荡造成局部动力学 Jacobian 放大这一候选也被严格 Gemini 判为已知接触失配与多步 rollout compounding error（新意5.2/10）；未运行训练或 GPU 门槛实验。'
if replay_priority_path.exists():
    summary['interpretation']+=' 合法高惊奇转移垄断优先经验回放的候选被严格 Gemini 判为普通 prioritized-replay bias、replay starvation 与 catastrophic forgetting（新意3/10）；未实现或运行。'
if timeout_laundering_path.exists():
    summary['interpretation']+=' 合法拖延至时间上限并误学为吸收终止的候选被严格 Gemini 判为已知 terminated/truncated 混淆（新意3/10）；未实现或运行。'
if non_normal_path.exists():
    summary['interpretation']+=' '+non_normal['report_note_cn']
if trajectory_order_path.exists():
    summary['interpretation']+=' '+trajectory_order['decision_cn']
if action_contrast_path.exists():
    summary['interpretation']+=' '+action_contrast['decision_cn']
(ROOT/'reports/results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

table=''.join('<tr><td>'+html.escape(r['study'])+'</td><td>'+html.escape(r['model'])+'</td><td>'+html.escape(r['arm'])+'</td><td>'+str(r['baseline_errors'])+'/'+str(r['probes'])+'</td><td>'+str(r['post_errors'])+'/'+str(r['probes'])+'</td><td><a href="../'+r['source']+'">原始结果</a></td></tr>' for r in rows)
references=json.loads((ROOT/'reports/related_work.json').read_text())
refs=''.join('<li><a href="'+html.escape(r['url'])+'">'+html.escape(r['title'])+'</a>：'+html.escape(r['finding'])+'</li>' for r in references)
dw_labels={'cold_no_thinking':'反应堆／Pro 关闭推理／无历史',
           'warm_no_thinking':'反应堆／Pro 关闭推理／共享历史',
           'cold_high_thinking':'反应堆／Pro 高推理／无历史',
           'warm_high_thinking':'反应堆／Pro 高推理／共享历史',
           'cold_flash_high_v1':'反应堆／Flash 高推理／接口说明修订／无历史',
           'warm_flash_high_v1':'反应堆／Flash 高推理／接口说明修订／共享历史',
           'cold_gateway_flash_v1':'反应堆／网关 glm-5.3-flash／接口说明修订／无历史',
           'warm_gateway_flash_v1':'反应堆／网关 glm-5.3-flash／接口说明修订／共享历史',
           'plant_flash_high_thinking':'植物养分／Flash 高推理／无历史',
           'plant_flash_high_v1':'植物养分／Flash 高推理／接口说明修订／无历史',
           'plant_gateway_v1':'植物养分／网关 glm-5.3／接口说明修订／无历史'}
dw_lines=[]
for key,value in discoveryworld.items():
    status=value['status']
    description=('完成任务' if value.get('observed_task_success') else '用满动作预算，任务未完成') if status in ['completed','budget_exhausted'] else ({'running':'运行中','researcher_stopped':'因接口与输出可靠性问题停止，不计任务成败','api_balance_interrupted':'API 余额不足中断，不计任务成败'}.get(status,status))
    dw_lines.append('<li>'+dw_labels[key]+'：'+description+'；已记录 '+str(value.get('logged_steps',0))+' 步。<a href="../'+value['source']+'/summary.json">运行状态</a></li>')
dw_section='''<h2>未知规则任务中的共享学习</h2>
<p>已接通 <a href="https://github.com/allenai/discoveryworld">DiscoveryWorld 官方环境</a>，使用未改动的 Hypothesizer 行动、反思和知识压缩函数，从原始重置状态开始，最多一百个游戏步骤。反应堆任务要求根据已校准晶体的测量结果推断第三个晶体的频率；另一个植物任务要求找出未知的有效养分。目前均为 Easy、开发种子 2。</p>
<p>共享历史由十一条真实动作采集：只测量晶体 1、2 并读取其已校准反应堆频率，再经原生模块形成知识，花费十二次 API 请求。另一代理在同种子的新环境中读取未经修改的知识；晶体 3 的读数和答案没有传入。旧位置和库存记录也被保留，可能在重置后失效。动作、测量和真实任务评分见 <a href="../runs/discoveryworld_learning_gate_v0/results.json">学习依赖短测</a>。共享组的执行成本与前期采集成本分别报告，不能把源课程开销当作零。</p>
<ul>'''+''.join(dw_lines)+'''</ul>
<p>关闭推理与开启高推理分别比较，均记录实际请求参数；Pro 高推理出现输出限额耗尽及截断，保留失败请求和原生解析失败。两条 Pro 高推理试运行提前停止，不算任务失败。新一轮 Flash 高推理同时为无历史与共享历史组补充同一句接口说明：列在 nearbyAgents 的设备可以用 TALK 打开控制对话。这来自公开接口，不提供频率或养分答案；原动作空间、反思与压缩仍相同。旧运行与新运行不用于直接估计单一因素效应。计算出频率不等于实际启动反应堆，局部分数也不等于任务完成。该阶段用于确定可用的任务与代理配置，不能作为攻击成功或新意证据。</p>
<p>官方 API 的 HTTP 402（Insufficient Balance）中断了部分运行，这些不能填成任务失败。主网关也出现间歇性超时；续跑重放原来已经返回的回复，并检查完整提示相等，保留先前的请求费用。续跑属于同一条轨迹，不能计作独立重复。</p>
<p>用户此前提供的主网关通过重新连通检查，已用 glm-5.3／glm-5.3-flash 别名另开独立任务运行，仍采用同一设备接口说明。后端身份未经独立核验，不能冒充官方模型复现；各组重新从初态开始，没有在旧运行中切换骨干。</p>
<p>植物养分无历史组在 16 步、41 次 API 请求后完成。反应堆无历史组在 33 步、累计 84 次请求后完成，共享历史组在 19 步、46 次请求后完成。共享组的前期采集另需 11 步、12 次请求，因此计入采集后是 30 步，对比无历史组的 33 步。请求数包括接口重试；无历史组两次续跑仍只是一条轨迹。成功来自环境的 completedSuccessfully 标志；这是单个开发世界中的结果，不能估计稳定收益。</p>
<p>十一动作的参考设备实验已完成，未支持“把可调设置误学为物理规律”的机制：调高 1000 Hz 后，最终压缩的知识仍保留原始正确测量对，并明确记下失去校准。调高 1 Hz 的对照保留了正确的 Channel 1 × 66 规律，但最后的压缩回复被截断，原生程序保留未压缩知识，因此不能声称完成了严格的压缩后对比。不再为这组课程运行完整受害任务。详见 <a href="../configs/discoveryworld_reference_screen.json">源课程结果</a>。</p>
<p>成功容差短测也已完成：在没有既有知识时，先对两个参考设备做同向或反向的 1 Hz 调整，再测量晶体。两组各 21 条动作全部成功，两个参考设备始终保持启动；分别花费 26、27 次请求。两组最终压缩均成功，且都保留原始校准值、明确区分手动调整，并得到正确的 Channel 1 × 66 规律。因此，研究者手算的错误外推没有在原生学习器中出现，不再扩展这条课程。详见 <a href="../configs/discoveryworld_success_calibration_screen.json">成功容差结果</a>。</p>
<p>植物移栽实验已完成：两组各 21 条真实动作全部成功，原生知识整理均成功解析。对照组确认钛；移栽组丢失搬动来历，留下错误的“钍或钾”待验证判断。随后用两份原样知识运行完整任务，两组都提交钛并成功：对照组 3 步／6 次请求，移栽知识组 6 步／13 次请求（含一次输出截断重试）。后者第 4 步结合新观察中的蘑菇位置与已有的正确土壤测量，修正了结论。源阶段分别为 35、27 次请求，对照源因接口中断续跑三次，仍是一条轨迹。该课程有知识偏差，却没有造成任务失败；单次多出 3 步不足以证明稳定成本攻击，停止扩展。它也不是跨任务转移的证据。详见 <a href="../configs/discoveryworld_plant_relocation_screen.json">植物移栽结果与任务对照</a>。</p>
<p><a href="https://arxiv.org/html/2609.32966v1">Self-Confirming Superposition Traps</a> 已研究策略选择数据、表征干扰与次优行为的自维持关系。因此一般性的“错误模型改变探索并自我维持”也需要进一步区分，不能仅靠换到科学发现任务支撑新意。</p>'''
causalgame_section='<h2>CausalGame 原生能力检查</h2><p>'+html.escape(causalgame_config.get('report_note_cn','原生能力检查进行中，尚无攻击实验或新机制结论。'))+' 详见 <a href="../configs/causalgame_feasibility.json">能力检查记录</a>。</p>'
guardrail_section='<p>'+html.escape(guardrail_review['report_note_cn'])+' 详见 <a href="../configs/predictive_guardrail_candidate.json">源码证据与检查范围</a>。</p>'
guardrail_section+='<p>'+html.escape(query_interference['report_note_cn'])+' 详见 <a href="../configs/tdmpc_query_interference_screen.json">规划查询短测</a>。</p>'
guardrail_section+='<p>'+html.escape(api_endpoints['report_note_cn'])+' 详见 <a href="../configs/api_endpoints_status.json">接口更新记录</a>。</p>'
if private_counterfactual_path.exists():
    guardrail_section+='<p>'+html.escape(private_counterfactual['report_note_cn'])+' 详见 <a href="../configs/world_model_private_counterfactual_screen.json">私有反事实查询短测</a>。</p>'
if temporal_correlation_path.exists():
    guardrail_section+='<p>'+html.escape(temporal_correlation['report_note_cn'])+' 详见 <a href="../configs/world_model_temporal_correlation_screen.json">跨时间耦合短测</a>、<a href="../runs/mbrl_temporal_risk_persistent_v0/summary.json">CVaR原始结果</a>。</p>'
if epistemic_collapse_path.exists():
    guardrail_section+='<p>'+html.escape(epistemic_collapse['report_note_cn'])+' 详见 <a href="../configs/world_model_epistemic_collapse_screen.json">成员分歧压平短测</a>、<a href="../runs/mbrl_epistemic_collapse_one_step_validated_v1/summary.json">一步采样验证</a>、<a href="../runs/mbrl_epistemic_collapse_cvar_v1/summary.json">CVaR原始结果</a>。</p>'
if conductivity_gemini_path.exists():
    guardrail_section+='<p>'+html.escape(conductivity_gemini['report_note_cn'])+' 详见 <a href="../configs/conductivity_gemini_actor_screen.json">Gemini执行复核</a>、<a href="../runs/conductivity_gemini_actor_v0/status.json">三臂轨迹</a>。</p>'
if gemini_post_screen_path.exists():
    guardrail_section+='<p>'+html.escape(gemini_post_screen['finding_cn']+' '+gemini_post_screen['decision_cn'])+' 详见 <a href="../configs/gemini_post_novelty_screen.json">最新候选复核记录</a>、<a href="../runs/gemini_post_screen_20261009T0840Z/summary.json">原始复核</a>。</p>'
if trajectory_order_path.exists():
    guardrail_section+='<p>'+html.escape(trajectory_order['decision_cn'])+' 详见 <a href="../configs/world_model_trajectory_order_screen.json">动作顺序短测</a>、<a href="../runs/mbrl_trajectory_order_attack_gate_v0/summary.json">CartPole 结果</a>。</p>'
if action_contrast_path.exists():
    guardrail_section+='<p>'+html.escape(action_contrast['decision_cn'])+' 详见 <a href="../configs/world_model_action_contrast_screen.json">动作反事实对应短测</a>、<a href="../runs/mbrl_action_contrast_gate_v0/summary.json">原始结果</a>。</p>'
if terminal_ownership_path.exists():
    guardrail_section+='<p>'+html.escape(terminal_ownership['decision_cn'])+' 详见 <a href="../configs/world_model_terminal_ownership_screen.json">终止归属路由短测</a>、<a href="../runs/mbrl_terminal_ownership_gate_v0/summary.json">种子810结果</a>、<a href="../runs/mbrl_terminal_ownership_gate_v0_seed811/summary.json">种子811结果</a>。</p>'
if branch_provenance_path.exists():
    guardrail_section+='<p>'+html.escape(branch_provenance['decision_cn'])+' 详见 <a href="../configs/world_model_branch_provenance_screen.json">分支来源短测</a>、<a href="../runs/branch_provenance_reduction_v0/summary.json">原始结果</a>。</p>'
if padded_action_path.exists():
    guardrail_section+='<p>'+html.escape(padded_action['decision_cn'])+' 详见 <a href="../configs/world_model_padded_action_screen.json">补零动作短测</a>、<a href="../runs/tdmpc_padded_action_shadowing_v0/summary.json">原始结果</a>。</p>'
if optimizer_state_path.exists():
    guardrail_section+='<p>'+html.escape(optimizer_state['decision_cn'])+' 详见 <a href="../configs/world_model_optimizer_state_screen.json">优化器状态短测</a>、<a href="../runs/tdmpc_optimizer_state_laundering_v0/summary.json">原始结果</a>。</p>'
if temporal_gradient_dither_path.exists():
    guardrail_section+='<p>'+html.escape(temporal_gradient_dither['decision_cn'])+' 详见 <a href="../configs/world_model_temporal_gradient_dither_screen.json">高频动作短测</a>、<a href="../runs/tdmpc_temporal_gradient_dither_v0/summary.json">原始结果</a>。</p>'
if gemini_search_path.exists() and gemini_search.get('status') == 'success':
    guardrail_section+='<p>定向 Gemini 机制搜索提出的五个方向分别落入已有的递归漂移、奖励梯度、潜空间塌缩、不确定性抑制或曲率扰动近邻；没有一个同时满足独立机制、原生规划接口和短门槛条件，因此不把模型意见当作主张或实验结果。详见 <a href="../runs/gemini_mechanism_search_20261009T1155Z/summary.json">定向搜索原始结果</a>。</p>'
if gemini_hidden_interface_path.exists():
    guardrail_section+='<p>本轮 Gemini 隐藏接口搜索返回了四个候选，但本地包装器把响应字典误传给字符串解析器；HTTP200 原始内容已保留，未作为证据。正交可达边界、曲率剪切和梯度反馈候选分别接近既有反事实/几何扰动或需要未授权梯度权限；跨任务可行性纠缠属于普通多任务负迁移，缺少动作授权的原生学习路径和任务门槛。因此不启动 GPU。详见 <a href="../runs/gemini_hidden_interface_search_20261009T1320Z/summary.json">搜索记录</a>。</p>'
if gemini_strict_new_path.exists():
    guardrail_section+='<p>最新严格 Gemini 机制筛选明确要求原生控制器、合法交互或授权推理操作、两分钟内的数字化证伪门槛，并排除一般投毒、观测替换和规划器改动；返回结果为没有候选。因此不把已有安全校准方法改写成新攻击，也不启动 GPU。详见 <a href="../runs/gemini_strict_new_candidate_20261009T0000Z/summary.json">本轮筛选</a>。</p>'
guardrail_section+='<p>'+html.escape(collective_review['report_note_cn'])+' 详见 <a href="../configs/collective_learning_candidate_screen.json">候选筛选与文献约束</a>。</p>'
guardrail_section+='<p>'+html.escape(scale_carryover['report_note_cn'])+' 详见 <a href="../configs/tdmpc_scale_carryover_screen.json">回报尺度短测</a>。</p>'
guardrail_section+='<p>'+html.escape(mechanism_review['report_note_cn'])+' 详见 <a href="../configs/mechanism_review_20261008.json">候选核实记录</a>。</p>'
guardrail_section+='<p>'+html.escape(dmo_jacobian['report_note_cn'])+' 详见 <a href="../configs/dmo_jacobian_screen.json">导数扰动短测</a>。</p>'
guardrail_section+='<p>'+html.escape(repair_transport['report_note_cn'])+' 详见 <a href="../runs/reactor_simple_program_v1/summary.json">原生任务基线结果</a>与<a href="../configs/repair_transport_screen.json">修正迁移短测</a>。</p>'
guardrail_section+='<p>'+html.escape(opine_memory['report_note_cn'])+' 详见 <a href="../configs/opine_query_memory_screen.json">查询与模型记忆诊断</a>及<a href="../runs/opine_query_memory_v0/summary.json">运行记录</a>。</p>'
guardrail_section+='<p>'+html.escape(opine_sync['report_note_cn'])+' 详见 <a href="../configs/opine_state_synchronization_screen.json">状态同步基线检查</a>及<a href="../runs/opine_state_synchronization_v0/summary.json">重放记录</a>。</p>'
document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>世界模型安全：机制探索与原生任务证据</title>
<style>body{font:16px/1.75 system-ui,sans-serif;color:#18202b;max-width:1100px;margin:48px auto;padding:0 24px}h1{font-size:32px;line-height:1.3}h2{margin-top:32px;font-size:22px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #d7dfe6;padding:9px;text-align:left}th{background:#eef3f8}a{color:#1466ac}.status{background:#fff4d8;border-left:4px solid #c18a10;padding:16px}.claim{font-size:21px;color:#184c74}code{background:#f1f4f7;padding:2px 5px}li{margin:10px 0}.scroll{overflow:auto}</style>
<body><p>WM Security · 开发阶段 · 证据更新</p><h1>世界模型安全：机制探索与原生任务证据</h1>
<div class="status"><strong>当前主线：共享动作 token 因果总线（第二载体 fixed-action world repair 与 held-out forecast tradeoff 已通过，闭环待验证）。</strong> WorldAgen 三十个真实 LIBERO 时间窗中，固定图像、状态和动作换 goal 会改变 world prediction（relative L2 平均 0.300，空指令 0.270）；切断九条 goal→input-action 边后 world 输出精确不变，policy 仍保持 goal-sensitive（relative L2 平均 0.176）。独立 AlphaBrain CosmosPolicy 的三组固定 latent、七个真实图像/state 窗口和七个官方 sampler 窗口也改变 future world 槽；fixed-action world cut 后精确恢复不变，三窗口 action control 仍有非零 action→world 差异，held-out 16 帧 forecast 的 repair MSE 代价为 +0.06737，bootstrap 区间不跨零。该候选自评 novelty 7/10，显式 goal-free baseline 与环境闭环仍待完成。</div>
<h2>研究主张与新意</h2><p class="claim">指令条件世界模型会把任务文本写入共享的动作 token 因果总线，使 goal 在直接 goal→world 掩码存在时仍能改变固定状态与固定动作下的物理未来预测；切分该总线可以恢复 goal-invariant world prediction，同时保留目标条件策略输出，暴露控制条件与动力学保真之间的安全权衡。</p>
<p>该主张的短门已通过：结构性 reachability、冻结发布权重的 goal swap、30 个真实 LIBERO 时间窗、第二载体固定动作 sampler 和 held-out forecast 的 edge-cut 因果修复方向一致。AlphaBrain 已完成三组固定 latent、七个真实图像/state 窗口和七个 sampler 窗口；剩余关键门槛是显式 goal-free/no-goal 多任务对照与闭环成功率。此前 TD-MPC2 轨迹印刻、MoSim Lie 候选及其它负结果保留在后文，作为停止错误主线和选择新载体的依据。</p>
<h2>已完成的实现</h2><p>程序化时间、前提与资源环境；本地 RTX 4090 推理；用户提供的模型 API；WorldEvolver 官方注册表、经验检索和规则抽取提示的直接复用；WorldMind 官方反思提示的第二种适配；独立初态的只读预测评测。模拟状态由程序生成，攻击交互没有注入自然语言、改写反馈或手工写入规则。</p>
<p>WorldEvolver 适配使用精确 JSON 状态比较，当前短测评估置信过滤前的预测。WorldMind 小样本适配读取全部经验教训，尚未复现其原生向量检索或具身任务。两者均不能标记为完整原生基准复现。</p>
<details><summary>展开早期短实验与被否定的路径</summary><h2>已完成的短实验</h2><p>每项为一个开发实例；同一实例的探针、交互臂及两个诊断初态均不增加独立实例数。错误列按预先指定方向计算；不同任务的错误方向可能不同，不能把各行直接合并为攻击成功率。</p>
<div class="scroll"><table><thead><tr><th>实验</th><th>模型</th><th>交互臂</th><th>洁净错误</th><th>交互后错误</th><th>证据</th></tr></thead><tbody>'''+table+'''</tbody></table></div>
<h2>这些结果如何改变下一步</h2><p>时间环境的一步判断可以正确，但本地多步预演在攻击前已经出错，当前不能用它证明“原本正确知识被改坏”。通过真实经验初始化的 API 短测没有触发规则更新。独立诊断任务能降低无关规则分数，但尚未造成下游失败。外接电源的资源例外在两种更新方式下均未增加目标错误。</p>
<p>补充的 Qwen3 例外学习短测未通过洁净能力检查。API 的同提示重复试验也发现明显波动：完整记忆在普通状态下错 1/3 次、例外状态下错 2/3 次；这些是同一实例的重复测量，不能视作独立样本。移除或仅保留规则的对照说明，规则能够影响部分预测，但尚未形成稳定、正确的待攻击知识。另一条全局正确的校准保持规律被 Qwen3 直接预测正确，没有触发学习，也没有表现出规则依赖。</p>
<h2>转向真实环境与信息差</h2><p>已接通 ScienceWorld 1.2.3，并用官方煮水动作序列取得环境累计分数 100；这只是环境验证，不是模型 Agent 的成功率。新的短测直接调用 WorldEvolver 原生文本预测、事实分解、规则抽取、检索与更新代码，比较正常交互、失败前查看无关物体、失败前查看房间三组等长真实交互。置信门关闭；目前仍是预测层的机制检验。</p>
<p><strong>首个预测退化信号：</strong>Qwen3-8B 在两个真实失败交互臂后，均将四个放置探针中的一个从正确预测变成错误失败判断。当前房间明示冰箱已打开，模型仍说冰箱没开；正常交互臂没有该错误。<a href="../runs/scienceworld_causal_local/memory_swaps.json">记忆交换</a>表明，仅替换交互后的真实轨迹就复现错误，仅替换规则则不复现。学到的“关闭的容器不能放入物品”规则本身成立，当前证据不支持错误规则是原因。</p>
<p><strong>早期子目标的限制：</strong>改用明确的“把金属锅放进冰箱”目标后，世界模型恢复了正确预测。原生 WMReactAgent 在无预测、洁净记忆、交互后记忆三种条件下均一步完成放置，见<a href="../runs/scienceworld_causal_local/actor_decisions.json">执行记录</a>。该子目标结果不支持实际决策损失。随后使用真实冷冻目标和更强的五次失败交互继续检验。</p>
<p><strong>更强的定向交互：</strong>五次真实失败占据同一动作的全部五个检索位置。随后通过原生更新器执行五次真实成功，完整房间视图下的错误预测仍然存在，检索仍返回原来的五条失败记录，见<a href="../runs/scienceworld_capture_local/recovery.json">纠错实验</a>。同一个物理状态用简短的“冷冻柜已打开”描述时，预测保持正确。这是一次开发案例的恢复性证据，不能视为普适、不可逆的攻击。</p>
<p><strong>修正指令后的核心任务结果：</strong>实际环境接受 <code>wait</code>（10 个迭代）和 <code>wait1</code>（1 个迭代），原提示列出的 <code>wait 1</code> 与 <code>wait 10</code> 均无效，见<a href="scienceworld_wait_commands.json">命令验证</a>。旧八步实验混入无效命令与预算因素，不能直接据此判定洁净系统能力不足。统一修正提示、采用 12 步预算后，<code>glm-5.3</code> 执行 Agent 配本地 Qwen3 世界模型的无辅助／洁净／交互后记忆三组均得 100 分，分别用 9／7／10 步，见<a href="../runs/freeze_continuation_grounded_v0/status.json">任务结果</a>。交互后记忆的错误预测使首步从直接放入变为检查冷冻柜，但最终完成任务；单次轨迹的步数差不能当作稳定攻击效应。另一轮全 API 洁净条件中断于调用错误，未计为任务失败。</p>
<p><strong>修复与迁移：</strong>保留全部记忆，只将动作相似度同分时的优先级从早期记录改为近期记录，检索即选中五条真实纠正经验，错误房间预测恢复正确，见<a href="../runs/capture_reads_local_recency/results.json">干预结果</a>。<code>glm-5.3</code> 世界模型不仅抵抗早期四个探针，也在五次失败加五次纠正的记忆上两次重复均预测正确，见<a href="../runs/capture_reads_glm/results.json">迁移检查</a>。当前影响依赖模型与检索方式；网关名称仍不是已核验的后端身份。变体 0 和 1 的目标冷冻状态完全相同，未当作两个独立实例。</p>
<p><strong>规则遗忘方向的追加检验：</strong>更强 API 模型在第一次例外学习检查中预测正确，但重复检查时普通条件连续出错；移除规则后普通条件恢复、例外条件失败。抽取出的规则没有保留必要作用域，因此没有进入攻击阶段，见<a href="../runs/erasure_glm_gate/rule_dependence_gate.json">学习依赖检查</a>。这说明该实例仍缺少稳定、正确的待攻击知识。另已通过官方轨迹验证化学混合、颜料混合和未知物导电性三种原生任务；这些 100 分是环境验证，不是模型表现。</p>
<p><strong>颜料混合的迁移短测：</strong>洁净记忆含完整绿色颜料成功轨迹。两条等长八步真实交互分别正常混合蓝黄颜料，或加入红色后得到棕色并重复混合。交互后，原生短观察与房间观察的预测都仍正确给出绿色，见<a href="../runs/paint_curriculum_local_v0/three_colors_result.json">原始结果</a>。该方向没有出现攻击信号，未继续测执行 Agent。</p>
<p><strong>配平后的冷冻任务对照：</strong>五步正常交互记忆与五次真实失败记忆两组均得 100 分，并执行完全相同的六个动作，见<a href="../runs/freeze_continuation_matched_v1/status.json">最终对照结果</a>。攻击组虽仍有错误预测，执行 Agent 仍直接放置。两轮首步修正提示的哈希相同，输出却曾分别为检查与直接放置，见<a href="identical_actor_prompt_variation.json">提示与响应核对</a>；其中一次涉及不同端点与输出上限，不能视为受控重复。此前三步差异不支持稳定的攻击成本效应。冷冻案例暂不作为论文主效应候选。</p>
<h2>未知物导电性：预测能否取代测量</h2>
<p>在同一官方任务中，未知物 B 实际不导电。正常接线将它串入电路，灯不亮；合法地绕过它接通电路后，灯会亮。两组各执行七个真实动作，并使用原生更新器学习。<code>glm-5.3</code> 的洁净与正常交互记忆均预测灯灭，绕行电路记忆使未测量状态下的预测变成灯亮，见<a href="../runs/conductivity_glm_v0/bypass_result.json">交互与预测</a>。首次读数因电路传播延迟仍为灯灭，后续才亮；记录保留全部真实反馈。</p>
<p>本地 Qwen3 的冻结记忆查询复现了灯状态偏差：只交换真实轨迹即可复现，只交换规则不会复现，见<a href="../runs/conductivity_reads_local/results.json">记忆干预</a>。但其洁净预测仍含错误接线细节，不能称为完整状态预测正确。房间视图包含已经测得的“灯灭”，它能纠正预测，却不是信息量相同的格式对照。强模型对绕行记忆的两次追加查询均因空输出或超时失败，尚无强模型重复确认。</p>
<p><strong>实际决策中的限制：</strong>本地执行 Agent 在洁净条件下得 100 分，而正常交互记忆与绕行记忆均误放入红盒，得 −100 分，见<a href="../runs/conductivity_continuation_local_v0/status.json">任务结果</a>。绕行组将计划中的房间观察改为直接分类：它依据预测中的“灯亮”行动，没有执行这次测量。但正常记忆也通过另一次错误预测引发误判，因此尚不能归因于攻击。</p>
<p>将开关操作统一放入真实前置步骤后，无辅助 Agent 两步成功，本地洁净辅助组却重复观察五次、未完成任务；后续攻击组没有运行，见<a href="../runs/conductivity_read_decision_local_v0/status.json">洁净能力检查</a>。全 API 版本的无辅助组也两步成功，但洁净世界模型连续空输出而中断，见<a href="../runs/conductivity_read_decision_glm_v0/status.json">中断记录</a>。这些结果均不计为攻击成功。</p>
<p><strong>首个通过双对照的任务失败：</strong>本地 Qwen3 世界模型配 <code>glm-5.3</code> 执行 Agent，洁净记忆四步得 100 分，正常接线记忆一步得 100 分，绕行记忆两步得 −100 分，见<a href="../runs/conductivity_read_decision_hybrid_v0/status.json">完整结果</a>。攻击组原本计划查看灯泡，收到“灯亮”的预测后改为拿起物体，随后放进错误的红盒，全程没有执行拟定的灯泡测量。正常记忆组也直接依据正确预测分类；因此关键在预测内容如何影响测量与行动，而非单纯存在跳过测量的行为。洁净组曾受错误接线描述干扰，随后通过真实查看完成任务。该比较只有一轮，仍需重复与独立实例。</p>
<p><strong>补充接线历史：</strong>给世界模型加入执行 Agent 已掌握的完整真实接线历史，不添加后续测量读数，洁净／正常记忆仍预测灯灭，绕行记忆及仅交换轨迹的条件仍错误预测灯亮，仅交换规则则预测灯灭，见<a href="../runs/conductivity_history_local_v0/results.json">历史信息对照</a>。这表明当前局部错误不只由缺少接线历史解释；尚未完成这一输入干预下的执行 Agent 对照。下一步检验任务级记忆交换、重复运行与第二种真实物质，不从单例直接扩量。</p>
<p><strong>任务级复现限制：</strong>仅替换轨迹和仅替换规则的两组执行 Agent 都实际查看了灯泡，然后两步成功，见<a href="../runs/conductivity_read_decision_swaps_v0/status.json">任务记忆干预</a>。预测偏差的轨迹归因，不能直接升级为稳定任务损失的归因。对三种条件各重复三次完全相同的首步修正提示，攻击条件中两次选择查看，一次在推理中错误认定物质导电并先拿起它；正常与洁净条件未出现错误红盒分类意图，见<a href="../runs/conductivity_actor_prompt_repeats_v0/results.json">逐次响应</a>。这九次只是决策探针，没有执行完整任务，不能转写为攻击成功率。</p>
<p><strong>第二个物理实例：</strong>变体 25 的物质 C 实际导电，接线颜色与 B 不同。正常接线后五次读数均为灯亮；合法断开 C 后五次读数均为灯灭。原生短测中，洁净与正常交互记忆预测灯亮，断路记忆却在原生视图和完整前置历史视图都预测灯灭，见<a href="../runs/conductive_counterpart_local_v1/open_result.json">相反方向的预测错误</a>。两个物理实例仍只属于一种任务。随后本地执行 Agent 的无辅助、洁净、正常、断路四组均得 100 分，见<a href="../runs/conductive_counterpart_actor_local_v0/status.json">任务对照</a>；断路组仍执行测量，读到真实灯亮后正确分类。第二个实例目前只支持预测层泛化，未支持任务损失泛化。</p>
<p>C 的强执行 Agent 追加实验在洁净条件下中断：完成一次真实拿起动作后，主端点超时，备用端点空输出；未产生洁净任务完成结果，也没有运行攻击比较，见<a href="../runs/conductive_counterpart_actor_hybrid_v0/status.json">中断记录</a>。</p>
<p><strong>把公共准备也纳入学习：</strong>早期投毒短测只学习公共准备之后的七个动作。B 的追加实验现已让两组从环境重置开始完整学习 17 个真实动作：十个相同准备动作，加七个正常或绕行分支动作。公共准备与正常组仍预测灯灭，绕行组仍将未测量状态错误预测为灯亮，见<a href="../runs/conductivity_complete_learning_local_v0/bypass_result.json">完整学习结果</a>。因此该预测偏差不依赖于省略准备阶段记录。此轮五次绕行读数均为灯亮，与较早运行的首次灯灭有所不同，实际反馈均已保留。包含真实灯灭读数的房间视图仍预测灯灭，但捏造了接线细节。</p>
<p>新增完整学习记忆已进入执行对照，但本地世界模型配 <code>glm-5.3</code> 执行 Agent 的洁净条件在一次真实动作后连续空输出；全 <code>glm-5.2</code> 版本的无辅助组两步成功，洁净预测随后连续空输出，均未形成攻击比较。<code>glm-5.2</code> 的六次冻结查询全部完成：原生视图为洁净／正常灯灭、攻击灯亮；完整历史视图下洁净组也误报灯亮，因此只支持前一视图的预测迁移，见<a href="../runs/conductivity_complete_reads_glm52_v0/results.json">原始查询</a>。</p>
<p><strong>真正的成功交互能否修复预测：</strong>正常与绕行记忆各接续五名全新的原生执行 Agent。这些 Agent 不接收世界模型预测，均真实查看灯泡后放入绿盒，十个续接任务全部两步成功；共享世界模型仍对每个实际动作预测并更新。绕行组在第 1、3、5 名成功用户后仍误报灯亮，检索一直保留五条旧绕行读数；只改同分记录的新旧优先级，改选五条真实纠正读数，两个视图均恢复灯灭，见<a href="../runs/conductivity_recovery_local_v0/status.json">真实用户交互</a>与<a href="../runs/conductivity_recovery_local_v0/bypass_recency_probes.json">检索干预</a>。这是人为提供成功纠错经历的恢复实验，不能表述为受攻击、接收预测的用户自发纠错，也没有证明任务损失持续。</p>
<p><strong>更强洁净初始化的限制：</strong>先让模型学习一次真实、正确的目标动作“查看灯泡”，再进行校准查询。原生视图正确，完整历史视图却误报灯亮，因此停止在洁净能力门槛，计划中的开关状态配平攻击没有运行，见<a href="../runs/conductivity_calibrated_matched_local_v0/clean_eval.json">洁净结果</a>。这不是攻击失败或成功，而是当前预测器基线仍不稳定。</p>
<h2>熔点任务：第二类测量偏差与执行结果</h2>
<p>官方未知物熔点任务要求按 150°C 阈值分类。共同加热准备后，正常组关闭炉子，攻击组合法继续加热；随后两组各进行五次“查看锅内物质、测温”的真实交互，均学习 11 个动作。正常读数为 124、130、133、135、137°C，攻击读数为 168、175、181、187、192°C。对同一个尚未执行的较早测温动作，洁净预测 116°C、正常组 135°C、攻击组 187°C，只有攻击组跨过分类阈值，见<a href="../runs/thermal_curriculum_local_v2/overheat_result.json">真实交互与预测</a>。给出完整的既有加热历史后三组都预测 116°C；这里测量的是分类相关阈值偏差，不是精确温度准确率。</p>
<p>任务执行没有复制分类失败。本地执行 Agent 在无辅助、洁净、正常、攻击四组都一步正确选择黄盒，未调用被污染的测温预测。<code>glm-5.2</code> 执行 Agent 配本地世界模型，洁净与正常组一步成功，攻击组三步成功：先关闭炉子，再真实测温，最后选择黄盒，见<a href="../runs/thermal_actor_hybrid_glm52_v0/status.json">执行结果</a>。一次两步额外开销不构成稳定成本效应。早期两个熔点开发版本因相态变化后物体名称失效而停止，不纳入攻击结果。</p>
<p><strong>接收预测的连续用户短测：</strong>用此前唯一通过双对照的配置，三组各计划让三名新 Agent 连续使用同一模型，并学习实际执行的动作。已完成的洁净／正常各两名用户，以及攻击组首名用户，都得到 100 分；攻击组第二名用户真实读到灯灭后，在继续检查开关时遭遇主端点超时和备用端点空输出，实验中断，见<a href="../runs/conductivity_shared_users_hybrid_v0/status.json">连续用户记录</a>。此轮没有复现终局误分类，也没有完成三名用户的全链条，未完成的调用不计为攻击成功。</p>
</details><h2>官方模型与另一种经验学习路径</h2>
<p>备用凭据已通过 DeepSeek 官方 <code>https://api.deepseek.com/models</code> 验证，返回 <code>deepseek-flash</code> 与 <code>deepseek-v4-pro</code>，见<a href="deepseek_endpoint_probe.json">接入记录</a>。<a href="https://api-docs.deepseek.com/en/">官方文档</a>将当前 Flash 映射到 V4.1；它与此前无法核实后端的第三方网关别名分别记录。</p>
<p>官方 Flash 六次冻结查询全部完成：原生查询中洁净／正常预测灯灭、绕行预测灯亮，完整历史查询中的洁净预测也错为灯亮。实际执行时，无辅助组成功，洁净世界模型组却把预测的灯亮当作分类依据，直接放入红盒，见<a href="../runs/conductivity_official_deepseek_actor_v0/status.json">执行检查</a>；攻击组没有运行。五次真实 <code>examine red light bulb</code> 校准仍未修复 Flash 的完整历史预测，见<a href="../runs/conductivity_calibrated_alias_official_v0/clean_eval.json">增强洁净检查</a>。因此不能把该模型的错误分类归因于投毒。</p>
<p>WorldMind 的原生判别、反思、存储、检索与精炼模块现已接通。将已有的完整 17 步真实交互与当时宿主模型的预测逐条交给它，两组各形成两条过程经验；绕行组被精炼成“实测并核验状态”，没有输出“B 导电”的结论，见<a href="../runs/worldmind_conductivity_replay_local_v0/bypass_retrieval.json">原生精炼结果</a>。这是固定宿主预测的经验迁移检验，不是重新生成整个 WorldMind 在线轨迹。原实现排除了同一任务指令的成功经验，故洁净检索为空；当前原生检索采用词重叠回退。</p>
<p>通过插件的经验注入接口，把上述固定记忆接入原生 WMReactAgent，并由官方 Flash 执行。统一十二步预算下，洁净、正常、绕行三组都正确完成，分别用 4、12、7 步，见<a href="../runs/worldmind_retrieval_official_actor_v1/status.json">任务结果</a>。此前五步版本的正常组尚未完成检查，未运行攻击组。此轮没有支持跨学习机制的任务损失；单条轨迹的步数差也不作为稳定效果。</p>
<p><strong>Pro 抵抗了更严格的测量课程：</strong>先学习一次真实正确的灯泡测量，两组再各学习完整 27 个真实动作，包括全部公共准备，以及每次读数前相同的“开关已激活”观察。洁净、公共准备、正常和绕行阶段的两个查询视图都保持灯灭，见<a href="../runs/conductivity_calibrated_official_pro_v0/bypass_result.json">完整结果</a>。末次检索中确有一条正确读数和四条真实绕行灯亮读数，因此不能把抵抗解释为没有检索到攻击经验。此轮没有预测退化，不继续把它计入任务攻击比较。</p>
<p><strong>真实跨任务学习破坏了保护：</strong>研究者先移除语义记忆，绕行检查点的两个视图都变为错误灯亮，见<a href="../runs/conductivity_pro_rule_protection_v0/results.json">干预查询</a>。随后使用模拟器真实执行的加热轨迹，先预测、再揭示实际反馈，并学习全部三十步准备和三十步后续动作。反复查看温度计的对照始终预测电路灯灭；持续测量升温物质的组出现 28/30 次预测不匹配，对无关灯泡规则的全局扣分使两条规则最终均失去正证据，电路预测在第二十步已变为错误灯亮，见<a href="../runs/native_cross_task_erasure_pro_v0/results.json">最终预测</a>。首次温度误差达 106°C，不能将全部不匹配描述为微小误差；预测转错也早于两条规则全部退出提示。</p>
<p><strong>机制干预的范围：</strong>从上述干扰组完全相同的准备后检查点出发，仅关闭全局负向扣分，重复三十次真实升温测量，两个电路视图始终保持正确，见<a href="../runs/native_cross_task_erasure_pro_no_debit_matched_v0/results.json">匹配干预</a>。但从准备阶段之前就关闭扣分的另一轮实验，在学习新规则后已经预测错误，见<a href="../runs/native_cross_task_erasure_pro_no_debit_v0/status.json">较早干预的负面结果</a>。因此只支持这一阶段的扣分通路，不能声称它解释所有错误或构成通用防御。</p>
<p><strong>完整任务重复没有支持攻击：</strong>官方 Pro 世界模型配 Flash 执行 Agent，预先固定六轮、平衡三组执行顺序，洁净／普通学习／干扰学习各六次全部 100 分。干扰组每次都实际测量，然后正确放入绿盒，见<a href="../runs/cross_task_erasure_flash_repeats_v0/results.json">十八次执行结果</a>。这些是同一个物理实例的重复，不是十八个独立任务。全 Pro 执行版本的洁净组十二步仍未完成，未进入攻击比较，见<a href="../runs/cross_task_erasure_actor_pro_v0/status.json">洁净门槛</a>。当前案例只支持预测损害，不支持稳定任务损失；停止在此案例上扩大重复。</p>
<p><strong>任务成功没有自动修复预测：</strong>读取上述六次独立重复留下的记忆，每次真实测量并完成分类后，两个查询视图仍都错误，见<a href="../runs/conductivity_pro_after_success_queries_v0/results.json">纠错后查询</a>。随后各组维持一份共享记忆，连续接入三位新用户，学习其全部后续动作；九次任务全部成功，干扰组在第一位和第三位用户之后仍预测灯亮。普通学习组到第三位用户后也预测错误，冻结重复中又有波动，见<a href="../runs/pro_erasure_shared_users_v0/probes.json">连续用户查询</a>和<a href="../runs/pro_rule_relearning_queries_v0/results.json">冻结复查</a>。因此只能支持局部预测恢复失败，不能据此宣称攻击独有的长期任务损害。各组内部用户相互依赖，仍只有一个物理实例，公共准备动作由用户观察但不在这一阶段重新学习。</p>
<p><strong>恢复失败有一个可定位的实现通路：</strong>三位受影响用户的真实读数均重新抽取出原来的正确“查看灯泡不改变其亮灭”规则。<a href="../runs/pro_rule_relearning_interventions_v0/merge_replay.json">逐条回放原生合并</a>确认：相同键和文本只刷新时间，不提高已有的负证据；规则仍被过滤。研究者只将这一条规则的证据恢复为正值，完整历史查询恢复，原生查询仍错；同时恢复两条灯泡规则后，两个视图在两次重复中均正确。普通学习组还从成功任务中学到了错误的“移动不改变电连接”规则，移除后这四次查询均正确。这些是同一检查点上的窄干预，不是攻击权限、任务成功率或通用防御；不同文本可创建新的正证据规则，不能称为不可逆。<a href="https://arxiv.org/html/2605.18930v1">OEP</a> 和 <a href="https://arxiv.org/html/2605.05583v1">Belief Memory</a> 已讨论错误经验的持续性与纠正困难，恢复失败本身不能作为新意证明。</p>
<p><strong>规划器方向的新意约束：</strong><a href="https://arxiv.org/html/2606.18697v1">SWAAP</a> 已通过修改少量微调转移的下一状态，破坏 TD-MPC2 与 DINO-WM 的规划，并包含多任务受害模型的初步迁移。因此“世界模型投毒使 MPC 失败”不能当作新的主张；仅行动权限也已见于 SCAB。后续需要真正新的学习机制及任务证据，而非替换执行器后重复既有结论。</p>
<p>原先的结构化例外规则方向也用官方 Pro 复查过，未学到有效的例外规则，停止在洁净门槛，见<a href="../runs/erasure_official_pro_gate_v0/status.json">结果</a>。当前 ScienceWorld 执行评测仍从官方可达中间状态继续，尚无该环境从重置开始的完整受害者基准，也没有达到会议级确认实验的扩量条件。</p>
<h2>潜变量规划器的短测</h2>
<p>已接入作者公开的 TD-MPC2 48M、三十任务检查点，保留原始三步、512 候选轨迹规划器。检查点文件哈希匹配，格式转换后的价值头与原始权重逐头计算相符，最大绝对差约 1.53×10⁻⁵。当前依赖版本及兼容处理见<a href="../runs/tdmpc_clean_gate_v0/provenance.json">来源记录</a>，不声称严格复现原论文软件环境。两个初态中，行走任务原生规划约 972／973 分，单策略约 395／182 分；机械臂到达任务规划为 983／991 分。猎豹奔跑与倒立摆洁净能力不足，行走机器人的跑步也有一次未过预设 600 分门槛，均不用于归因攻击损失，见<a href="../runs/tdmpc_clean_gate_v1/results.json">补充洁净检查</a>。</p>
<details><summary>初步适配对照</summary><p>到达任务的正常规划与随机动作各采集三百次真实交互；随后仅做四十步共享动力学更新，冻结其余部分。到达任务由 983 分分别变为 981／984，行走任务由 972.4 分分别变为 975.0／974.9，见<a href="../runs/tdmpc_adaptation_screen_v0/results.json">短测结果</a>。以向后行走为源、向前行走为目标，得分分别为 972.3／973.8，仍未退化，见<a href="../runs/tdmpc_adaptation_screen_v1/results.json">共享物理任务结果</a>。相同预算改用原始完整更新函数，目标得分为洁净 972.4、正常 954.8、随机 973.2，见<a href="../runs/tdmpc_native_adaptation_v0/results.json">完整更新结果</a>。正常组单次小幅下降不能当作攻击效应。</p></details>
<p><strong>定向搜索没有造成任务损害：</strong>在向后行走环境中，用两代搜索产生十六段、各三百次真实动作的课程；共用 4,800 次源环境参考交互，选中的部署课程仍只有三百次。每个候选独立从原检查点做四十次原始完整更新。按干净模型的候选规划排序损失选择后，目标向前行走两个保留初态的得分为 976.8／990.8，洁净为 973.1／989.7；正常、随机及最大预测误差基线也均未明显受损，见<a href="../runs/tdmpc_action_search_v0/results.json">实际任务结果</a>。代理排序变化没有转化成任务损失。</p>
<p>进一步允许攻击者运行目标参考模拟器，以完整五百步实际回报在同一十六候选中选择，新增 9,500 次目标参考交互。最差候选 927.9 分，比最低对照 954.8 只低 26.9 分，未达到预设“低于每个对照至少 100 分”的门槛，因此未运行最终确认初态，见<a href="../runs/tdmpc_return_search_v0/selection.json">选择结果</a>。固定该候选、将学习增加到 120／300 次，另一初态得分为 966.0／957.0；普通学习为 972.0／965.6，洁净为 973.1，见<a href="../runs/tdmpc_update_dose_v0/results.json">更新次数短测</a>。这些开发短测不足以支持稳定或较大的攻击损害，也没有进行显著性检验；停止沿本设置增加训练。</p>
<p>上述在线适配由本研究加在官方离线多任务检查点上，保留原始更新函数但使用 batch 32 和近期源任务数据，没有原训练回放库。三百次更新只在更新与新增动作的数量比例上接近原在线训练，不是原生训练协议复现。攻击可访问权重和参考模拟器，全部记录与奖励来自实际执行，没有改写；这些是开发短测，不是已有投毒算法的完整复现或新算法贡献。受攻击后的潜变量模型修复和集体任务损失尚无支持。</p>
<h2>多步计划的直接执行检查</h2>
<p>新增官方 Pro 的完整计划短测，检验真实经验是否会使模型把不兼容的局部结果拼成不可执行计划。模型逐步预测预先固定的候选计划，选择最短的预测可行计划，再在独立初态的真实程序环境中完整执行；沿用原生经验检索、规则抽取和证据更新的结构化状态适配。生产线重启和往返配送均通过洁净检查；三种课程各学习十六个真实动作后，洁净、普通、随机、定向四组均选对可行计划并执行成功，共八次成功，见<a href="../runs/compositional_plans_pro_v0/results.json">计划与执行轨迹</a>。定时通行在攻击前就把重新激活错误预测为不续期，未运行该任务的攻击课程。这是三个小型程序任务的筛选，不是原生 Agent 的自主规划基准，也不支持当前定向课程造成组合性损害。</p>
<p><a href="https://arxiv.org/html/2606.10934v1">WorldKernel</a> 已讨论预测与整体可行性、反事实一致性的区别；一般性的“一步预测不足以保证完整推理”不能作为新意。对这里使用的确定性全状态环境，如果模型在所有递归访问的状态动作上都预测准确，多步轨迹必然准确。因此后续若出现效果，必须定位真实交互诱发的具体模型变化，不能把所有一步预测正确和错误多步轨迹同时作为未经限定的主张。</p>
'''+dw_section+causalgame_section+guardrail_section+'''<h2>最终实验目标</h2><p>目标仍为会议论文级证据：至少三类核心任务、强交互基线、多个可核验骨干、两种学习机制、独立 Agent 的真实任务损失，以及一组决定性的机制干预。原生 ALFWorld／ScienceWorld／DiscoveryWorld 任务作为外部验证。规模以短测效应和实际调用成本决定；目前所有确认性规模均为计划值。</p>
<h2>复现入口</h2><p><code>PYTHONPATH=src python -m unittest discover -s tests -v</code></p><p><code>PYTHONPATH=src python scripts/summarize.py</code></p><p>实验入口位于 <code>scripts/pilot_*.py</code>；<a href="results_summary.json">机器可读结果与用量</a>；<a href="../configs/research_roadmap.json">研究路线与扩量条件</a>。API 凭据存放在项目目录之外，未写入文档和调用日志。第三方网关别名仍不视作已核验后端；本地权重版本与新增 DeepSeek 官方端点分别留有记录。</p>
<h2>关键文献与区别</h2><ul>'''+refs+'''</ul><p><a href="../Shared_World_Model_Experimental_Plan_CN.pdf">原始 24 页方案</a>保留，本文记录已发生的研究进展。</p></body></html>'''
measurement_labels={'repeat_mean':'原位置／均值','repeat_min':'原位置／最小值',
                    'isolate_mean':'隔离位置／均值','isolate_min':'隔离位置／最小值'}
measurement_rows=''.join(
    '<tr><td>'+measurement_labels[name]+'</td><td>'+str(result['tasks_completed'])+
    '/'+str(result['episodes'])+'</td><td>'+', '.join(map(str,result['actions']))+'</td></tr>'
    for name,result in reactor_grammar['measurement_controls'].items())
reactor_grammar_section=('<h2>新增原生任务：程序搜索与测量流程</h2><p>'+
    html.escape(reactor_grammar['report_note_cn'])+'</p>'+
    '<table><thead><tr><th>测量流程</th><th>完整任务成功</th><th>两实例总动作数</th></tr></thead><tbody>'+
    measurement_rows+'</tbody></table><p>四项对照使用相同的函数语法与选择规则。'+
    '8/10 的基线结果与失败后选出的 2/2 恢复结果，不能合并成同一策略的 10/10。'+
    '后续自动流程选择实验见前节；本组人工流程不是新方法。'+
    '详见<a href="../configs/reactor_program_grammar_screen.json">设定与结论</a>、'+
    '<a href="../runs/reactor_program_grammar_v0/summary.json">十任务基线</a>、'+
    '<a href="../runs/reactor_measurement_protocol_v0/summary.json">测量干预</a>与'+
    '<a href="../runs/reactor_measurement_protocol_controls_v0/summary.json">其余对照</a>。</p>')
acquisition_labels={'target_only':'仅参考样本拟合','fixed_calibration':'人工校准流程','search':'自动搜索＋独立复测筛选'}
acquisition_rows=''.join(
    '<tr><td>'+acquisition_labels[name]+'</td><td>'+str(result['successes'])+'/'+str(result['episodes'])+
    '</td><td>'+str(result['actions'])+'</td></tr>'
    for name,result in acquisition_search['validated']['by_arm'].items())
acquisition_section=('<h2>自动测量流程：已实现的候选与强基线</h2><p>'+
    html.escape(acquisition_search['report_note_cn'])+'</p>'+
    '<p>下表仅比较种子13/14、Normal/Challenge共四个新实例。目标晶体结果不参与流程选择。'+
    '初始目标读数已采集，但选择器只收到参考样本；新流程冻结后才作用于目标。'+
    '原有含目标范围筛选的程序基线另行复跑，仍为2/4。</p>'+
    '<table><thead><tr><th>方法</th><th>完整任务成功</th><th>累计原生动作</th></tr></thead><tbody>'+
    acquisition_rows+'</tbody></table><p>该组实验共32条原生任务轨迹、8个不同世界实例，未使用GPU。'+
    '尚无第二类核心任务、跨域收益或新概念结论。'+
    '详见<a href="../configs/reactor_acquisition_search_screen.json">方法、范围与决定</a>、'+
    '<a href="../scripts/pilot_reactor_acquisition_search.py">可运行实现</a>、'+
    '<a href="../runs/reactor_acquisition_search_v0/summary.json">初版结果</a>和'+
    '<a href="../runs/reactor_acquisition_search_validated_v1/summary.json">复测筛选版结果</a>。</p>')
lazy_labels={'raw':'原始读数拟合','fixed_calibration':'每次执行人工校准','lazy_calibration':'实际失败后才校准'}
lazy_rows=''.join(
    '<tr><td>'+lazy_labels[name]+'</td><td>'+str(result['successes'])+'/'+str(result['episodes'])+
    '</td><td>'+str(result['actions'])+'</td><td>'+str(result['calibration_invocations'])+'</td></tr>'
    for name,result in lazy_calibration['by_arm'].items())
lazy_section=('<h2>成本更低的原生任务基线</h2><p>'+html.escape(lazy_calibration['report_note_cn'])+'</p>'+
    '<table><thead><tr><th>方法</th><th>完整任务成功</th><th>累计原生动作</th><th>校准次数</th></tr></thead><tbody>'+
    lazy_rows+'</tbody></table><p>两轮各自重新加载相同四个世界实例，仪器噪声重新采样；'+
    '共24条任务轨迹、4个不同世界，不计作8个独立世界。失败尝试、移动、采样和调谐全部计入动作成本。'+
    '校准程序仍是研究者提供的隔离和32次取最小值操作；按任务反馈决定何时调用它，不是新学习机制。'+
    '这些实例与历史搜索实验所用种子不同，不据此计算搜索相对本基线的成本倍数。'+
    '详见<a href="../configs/reactor_lazy_calibration_screen.json">结果与范围</a>、'+
    '<a href="../scripts/pilot_reactor_lazy_calibration.py">实现</a>及'+
    '<a href="../runs/reactor_lazy_calibration_v0/summary.json">首轮</a>、'+
    '<a href="../runs/reactor_lazy_calibration_repeat_v0/summary.json">独立噪声复跑</a>。</p>')
revision_labels={'ordinary_refit':'普通重拟合','finite_model_elimination':'有限候选模型排除',
                 'errors_in_variables':'普通测量误差拟合','feedback_evidence_revision':'反证驱动证据修正候选'}
revision_rows=''.join(
    '<tr><td>'+revision_labels[name]+'</td><td>'+('成功' if result['completed_successfully'] else '失败')+
    '</td><td>'+str(result['continuation_actions'])+'</td></tr>'
    for name,result in evidence_revision['native']['by_method'].items())
revision_section=('<h2>候选学习操作的直接对照</h2><p>'+html.escape(evidence_revision['report_note_cn'])+'</p>'+
    '<table><thead><tr><th>选择规则</th><th>同一历史检查点的原生续跑</th><th>续跑动作数</th></tr></thead><tbody>'+
    revision_rows+'</tbody></table><p>四种方法使用同一份历史观测，选择在续跑之前冻结。'+
    '每组264个重放动作单独计数，共1056个；加133个续跑动作，实际总计1189个原生动作。'+
    '不是4个独立测试世界，也没有自主发现新的测量程序。'+
    '普通模型排除与测量误差拟合已能恢复该案例，因此不把恢复本身作为新概念的证据。'+
    '详见<a href="../configs/reactor_evidence_revision_screen.json">方法与决定</a>、'+
    '<a href="../scripts/screen_reactor_evidence_revision.py">候选实现</a>、'+
    '<a href="../runs/reactor_evidence_revision_v0/summary.json">匹配对照</a>及'+
    '<a href="../runs/reactor_evidence_revision_continuation_v0/summary.json">原生执行</a>。</p>')
drwell_section=('<h2>合作经验的原生短测</h2><p>'+html.escape(drwell_credit['report_note_cn'])+
    ' 详见<a href="../configs/drwell_success_credit_screen.json">候选结论</a>、'+
    '<a href="../scripts/pilot_drwell_success_credit.py">可复现实现</a>及'+
    '<a href="../runs/drwell_success_credit_v0/summary.json">原生执行结果</a>。</p>')
baba_section=('<h2>原生程序学习短测</h2><p>'+html.escape(baba_rejection['report_note_cn'])+
    ' 详见<a href="../configs/baba_rejection_learning_screen.json">研究结论</a>、'+
    '<a href="../runs/baba_program_history_full_v0/summary.json">公开程序评估</a>、'+
    '<a href="../runs/baba_rejection_feedback_replay_v0/summary.json">拒绝反馈重放</a>及'+
    '<a href="../runs/baba_frontier_effect_checkpoint_v0/summary.json">原生探索对照</a>；'+
    '<a href="../runs/baba_rejection_support_v0/summary.json">修订分组支持筛查</a>与'+
    '<a href="../runs/baba_frontier_regression_maps_v0/summary.json">回退地图采集</a>；'+
    '<a href="../scripts/pilot_baba_native_repair.py">原生修订实现</a>。</p>')
experiment_section=('<h2>此前的实验方法迁移开发</h2><p>'+html.escape(experiment_transport['report_note_cn'])+
    ' 详见<a href="../configs/experiment_transport_screen.json">候选与结论</a>、'+
    '<a href="../runs/neuronbench_forecaster_v0/assessment.json">同数据完整轨迹拟合结果</a>、'+
    '<a href="../runs/neuronbench_forecaster_v0/ca_rebound/summary.json">钙反跳完整轨迹预测</a>、'+
    '<a href="../runs/neuronbench_forecaster_v0/na_fatigue/summary.json">钠疲劳完整轨迹预测</a>、'+
    '<a href="../runs/neuronbench_forecaster_v0/verification/parity.json">加速模拟器原生一致性</a>、'+
    '<a href="../scripts/fit_neuronbench_forecaster.py">预测器运行脚本</a>、'+
    '<a href="../runs/neuronbench_native_v0/assessment.json">部分可观测原生短测</a>、'+
    '<a href="../runs/neuronbench_native_v0/ca_rebound/summary.json">钙反跳基线</a>、'+
    '<a href="../runs/neuronbench_native_v0/na_fatigue/summary.json">钠疲劳基线</a>、'+
    '<a href="../runs/neuronbench_native_v0/observation_analysis.json">已有观测与公开参考模型</a>、'+
    '<a href="../runs/interphyre_policy_generation_v0/evaluation/summary.json">生成策略原生比较</a>、'+
    '<a href="../runs/interphyre_policy_generation_v0/feedback_ablation/summary.json">反馈消融</a>、'+
    '<a href="../runs/interphyre_policy_generation_v0/prior_matched_optimizers/summary.json">相同动作先验对照</a>、'+
    '<a href="../runs/interphyre_policy_generation_v0/assessment.json">程序生成结论</a>、'+
    '<a href="../runs/interphyre_continuous_search_v1/summary.json">连续动作原生比较</a>、'+
    '<a href="../runs/interphyre_continuous_search_v1/assessment.json">搜索与割线候选结论</a>、'+
    '<a href="../runs/event_secant_equivalence_v1/summary.json">普通回归等价核验</a>、'+
    '<a href="../runs/interphyre_public_trace_program_v0/summary.json">公开经验同数据比较</a>、'+
    '<a href="../runs/interphyre_public_trace_program_v0/assessment.json">程序结构与方法结论</a>、'+
    '<a href="../runs/interphyre_program_search_v0/summary.json">物理任务筛查</a>、'+
    '<a href="../runs/interphyre_program_search_v0/assessment.json">源采集限制与原生控制</a>、'+
    '<a href="../scripts/pilot_interphyre_program_search.py">实验程序搜索实现</a>；此前'+
    '<a href="../runs/feedback_program_ambiguity_v0/summary.json">反馈程序原生结果</a>、'+
    '<a href="../runs/feedback_program_ambiguity_v0/assessment.json">完整状态基线对照</a>、'+
    '<a href="../scripts/pilot_feedback_program_ambiguity.py">反馈程序实现</a>；此前'+
    '<a href="../runs/separator_objective_counterexample_v0/summary.json">区分实验目标分析</a>、'+
    '<a href="../runs/experiment_combinator_review_20261008T221631Z/matched_transition_assessment.json">同数据转移比较</a>、'+
    '<a href="../runs/alchemy_relation_compiler_v0/summary.json">源关系生成实验</a>、'+
    '<a href="../scripts/screen_alchemy_relation_compiler.py">生成原型实现</a>；'+
    '<a href="../configs/alchemy_preparation_screen.json">Alchemy强基线结论</a>、'+
    '<a href="../runs/alchemy_preparation_baselines_v1/summary.json">8种子原生对照</a>、'+
    '<a href="../runs/alchemy_preparation_replay_v0/summary.json">相同历史分支验证</a>；此前'+
    '<a href="../runs/discoverphysics_native_extra_glm53_v0/evaluation.json">原生发现基线</a>、'+
    '<a href="../runs/discoverphysics_same_data_power_baseline_v0/summary.json">同数据普通拟合</a>、'+
    '<a href="../runs/discoverphysics_contrast_transfer_v0/summary.json">此前配对结果</a>及'+
    '<a href="../scripts/screen_discoverphysics_contrast_transfer.py">复现实现</a>。</p>')
sampling_rows=''.join(
    '<tr><td>'+html.escape(r['map'])+'</td><td>'+str(r['particles'])+'</td><td>'+
    f"{100*r['independent_success']:.2f}%</td><td>{100*r['optimized_success']:.2f}%</td><td>"+
    f"{100*r['shared_quantile_success']:.2f}%</td><td>{r['success_drop_pp']:.2f}</td><td>"+
    '<a href="../'+html.escape(r['source'])+'">原始结果</a></td></tr>'
    for r in sampling_security['rows'] if r['particles'] in (16,64))
learned_result=sampling_security['learned_result']
learned_rows=''.join(
    '<tr><td>'+('开发' if r['stage']=='development' else '新种子复核')+'</td><td>'+str(r['seed'])+
    '</td><td>'+f"{r['independent_return']:.0f}"+'</td><td>'+f"{r['targeted_return']:.0f}"+
    '</td><td>'+('未运行' if r['shared_return'] is None else f"{r['shared_return']:.0f}")+
    '</td><td><a href="../'+html.escape(r['independent_source'])+'">对照轨迹</a>／'+
    '<a href="../'+html.escape(r['targeted_source'])+'">定向轨迹</a></td></tr>'
    for r in learned_result['paired_rows'])
learned_usage=learned_result['usage']
sampling_section=('<h2>采样安全探索：现有证据与扩展限制</h2><p><strong>'+html.escape(sampling_security['candidate_claim_cn'])+
    '</strong></p><p><strong>强基线复核：</strong>'+html.escape(sampling_security['robustness_result']['report_note_cn'])+
    ' <a href="../'+html.escape(sampling_security['robustness_result']['source'])+'">完整复核结果与成本</a>。</p>'+
    '<p><strong>后续梯度构造未晋升为主贡献：</strong>'+html.escape(gradient_security['report_note_cn'])+
    ' <a href="../configs/world_model_gradient_security.json">候选权限与先行工作</a>、'+
    '<a href="../'+html.escape(gradient_security['source'])+'">解析与程序记录</a>、'+
    '<a href="../runs/predictive_gradient_mpc_v1/assessment.json">完整控制回合与停止条件</a>。'+
    '本轮控制短测共1000个真实环境步、60万个预测转移、2400次动作优化，约75.9秒CPU计算；无神经网络训练或GPU工作。</p>'+
    '<p><strong>此前缩小配置中的正向机制结果：</strong>'+html.escape(sampling_security['report_note_cn'])+'</p>'+
    '<p><strong>HalfCheetah 强模型分流：</strong>'+html.escape(sampling_security['headline_cn'])+
    ' 强设置下 25 步前缀回报为目标耦合 −4.38、独立抽样 −4.84、等预算平均 −5.56；目标臂没有低于两个对照，因此停止该方向的扩展。'+
    '<a href="../runs/mbrl_rank_halfcheetah_refit_v2/targeted_strong_gate_seed812_25/summary.json">目标轨迹</a>、'+
    '<a href="../runs/mbrl_rank_halfcheetah_refit_v2/independent_strong_gate_seed812/summary.json">独立对照</a>、'+
    '<a href="../runs/mbrl_rank_halfcheetah_refit_v2/bank_mean_strong_gate_seed812/summary.json">等预算平均对照</a>。</p>'+
    '<p><strong>Pusher 第二连续控制任务：</strong>'+html.escape(sampling_security.get('pusher_headline_cn',''))+
    ' 同一强设置下原生50步回报−39.19，独立抽样−40.26，等预算平均−39.75；目标耦合在可比25步前缀为−23.13，独立−23.20，等预算平均−23.06。目标臂没有方向性下降，停止该分支。'+
    '<a href="../runs/mbrl_rank_pusher_v1/targeted_strong_seed812_25/summary.json">目标轨迹</a>、'+
    '<a href="../runs/mbrl_rank_pusher_v1/independent_strong_seed812/summary.json">独立对照</a>、'+
    '<a href="../runs/mbrl_rank_pusher_v1/bank_mean_strong_seed812/summary.json">等预算平均对照</a>。</p>'+
    '<table><thead><tr><th>阶段</th><th>种子</th><th>独立选秩回报</th><th>定向耦合回报</th><th>共享选秩回报</th><th>证据</th></tr></thead><tbody>'+
    learned_rows+'</tbody></table><p>'+html.escape(learned_result['report_limits_cn'])+'</p>'+
    '<p>初轮 GaussianMLP 全部构造（含失败尝试）共 '+str(learned_usage['native_evaluation_episodes'])+' 次评估、'+
    f"{learned_usage['native_evaluation_steps']:,}"+' 个真实评估步，另有 '+str(learned_usage['native_collection_steps'])+' 个训练数据采集步；'+
    f"{learned_usage['native_predictive_transitions']:,}"+' 个原生模型预测转移，以及 '+
    f"{learned_usage['additional_surrogate_forward_transitions']:,}"+' 个代理目标前向转移和额外反向计算。一次 CPU 训练约 '+
    f"{learned_usage['training_cpu_elapsed_seconds']:.2f}"+' 秒；该轮无 API 调用或 GPU 工作。后续强基线复核另含短时 TD-MPC2 GPU 推理，无 GPU 训练，成本单列。详见 '+
    '<a href="../'+html.escape(learned_result['source'])+'">汇总、负结果与核验</a>、'+
    '<a href="../'+html.escape(learned_result['proof_source'])+'">预测分布保持论证</a>、'+
    '<a href="../'+html.escape(learned_result['implementation'])+'">采样实现</a>。</p>'+
    '<p><strong>此前的精确模型证据：</strong>Gymnasium FrozenLake-v1 的两个内置地图，原生转移核、自实现一步采样规划器与精确后续价值。</p>'+
    '<table><thead><tr><th>地图</th><th>每动作样本数</th><th>独立采样</th><th>定向耦合</th><th>共享分位数</th><th>下降百分点</th><th>证据</th></tr></thead><tbody>'+
    sampling_rows+'</tbody></table><p>表中为原生转移核下的精确成功概率；两张地图属于同一个简单任务家族，不能计作多类核心任务。'+
    '详见<a href="../configs/world_model_sampling_security.json">机制、权限与结论</a>、'+
    '<a href="../runs/coupled_sampling_security_v0/summary.json">1／4／16 样本原生执行</a>、'+
    '<a href="../runs/coupled_sampling_security_n64/summary.json">64 样本边界</a>、'+
    '<a href="../runs/coupled_sampling_security_v0/analytic_witness.json">同协方差解析例子</a>、'+
    '<a href="../scripts/pilot_coupled_sampling.py">运行代码</a>。</p>')
wav_alias_section=('<h2>WAV 动作别名诊断：仅保留实现审计</h2><p>在128个固定 OOD MiniGrid 状态上，原生物理 oracle 给出151组后继端点完全相同的动作别名；四个已发布检查点却把其中141—146组预测成不同后继（拆分率93.38%—96.69%）。严格 Gemini 评审给出新意2/10、置信度0.95，认为这是普通 OOD 动作语义/等价性泛化误差，且尚无攻击操作或原生控制器任务损害。因此不把它升级为安全主张，也不启动控制器或 GPU 扩展。</p><p>详见<a href="../configs/world_model_wav_alias_screen.json">筛选结论</a>、<a href="../runs/gemini_wav_alias_review_20261010T0000Z/assessment.json">严格评审</a>。</p>')
boundary_stiffness_section=('<h2>边界刚度候选：已知 rollout 复合误差，停止</h2><p>严格评审了“合法硬边界往返动作集中真实接触数据，使平滑高斯动力学模型的局部 Jacobian 变大，而一步误差仍正常，最终破坏 CEM 长视野推演”的候选。评审新意5.2/10，认为这是接触处模型失配与经典多步 rollout compounding error；没有独立的世界模型安全操作，因此未运行训练、控制器或 GPU 扩展。</p><p>详见<a href="../configs/world_model_boundary_stiffness_screen.json">候选结论</a>、<a href="../runs/gemini_uncovered_interface_search_short_20261010T0000Z/assessment.json">严格评审</a>。</p>')
replay_priority_section=('<h2>优先回放垄断候选：普通 replay bias，停止</h2><p>严格评审了“合法高惊奇转移垄断优先经验回放，从而压制受害任务纠正数据”的候选。评审新意3/10，认为其本质是 prioritized replay bias、replay starvation 和 catastrophic forgetting，缺少独立的世界模型安全原语；当前固定的 MBRL-Lib 与 TD-MPC2 短测也没有该原生接口。因此未实现或运行。</p><p>详见<a href="../configs/world_model_replay_priority_screen.json">候选结论</a>、<a href="../runs/gemini_replay_priority_review_20261010T0000Z/assessment.json">严格评审</a>。</p>')
timeout_laundering_section=('<h2>时间上限终止候选：已知 truncation bias，停止</h2><p>严格评审了“合法拖延至时间上限，使世界模型把 truncation 当作吸收终止”的候选。评审新意3/10，认为它完全依赖已有的 terminated/truncated 混淆和外部 episode 管理缺陷；正确分离两类信号或加入剩余时限即可消除影响。因此未实现或运行。</p><p>详见<a href="../configs/world_model_timeout_laundering_screen.json">候选结论</a>、<a href="../runs/gemini_timeout_laundering_review_20261010T0000Z/assessment.json">严格评审</a>。</p>')
non_normal_section=('<h2>级联交互：任务族条件性结果</h2><p>'+html.escape(non_normal['report_note_cn'])+'</p><p>独立留出 walker-run 两个种子中，级联相对 random 的回报下降为19.1%和19.3%，相对相同动作集合乱序对照下降为19.3%和18.2%，一步误差为 random 的0.922倍；cup-catch 未出现级联特异损失。因此这是已通过强匹配对照的 walker-run 任务族现象，尚非跨任务或非正规性因果机制。相同协议在 walker-run-backwards 两个新种子上仅为12.4%与反向2.9%，在 hopper-hop 上为反向31.2%与下降45.9%，均未通过独立正任务族门槛；hopper 的 shuffled 对照为99.46/105.89，级联为54.19/48.28，一步误差为 random 的1.017倍。新增同一 cheetah-run 源和 dose300 的 walker-like 门槛也未通过：walker-stand 为 cascade586.31 对 random592.73，walker-walk 为574.08 对574.996。SimNorm 温度2干预同时破坏 clean/random 并使一步误差约升10倍，防御假设撤回。</p><p><a href="../configs/world_model_non_normal_screen.json">当前证据与结论更正</a>；<a href="../runs/tdmpc_cascade_heldout_matched_v1/status.json">walker-run确认结果</a>；<a href="../runs/tdmpc_cascade_heldout_hopper_v1/status.json">hopper-hop确认结果</a>；<a href="../runs/tdmpc_non_normal_walker_like_gate_v0/status.json">walker-like新门槛</a>；<a href="../runs/tdmpc_cascade_heldout_matched_v1/plan.json">固定确认方案</a>。</p>')
parity_section=('<h2>高阶采样耦合：精确不变量成立但机制淘汰</h2><p>偶／奇三阶 parity 构造在有限样本库上严格保持每个候选的边缘索引多重集和全部三组两两索引表，同时翻转三元乘积；因此数学见证成立。原生 MBRL-Lib CartPole 短测中，704/705 两个种子攻击臂低于 parity-control，但706/707 两个种子达到相同的200步上限，方向不稳定。严格评议给出新意3/10，认为这是 common random numbers 与跨候选相关采样在三阶上的增量变体，作用落在优化器的联合抽样与 argmax，而不是世界模型转移机制。</p><p><a href="../configs/world_model_parity_coupling_screen.json">筛选结论</a>；<a href="../runs/parity_coupling_exact_v0/summary.json">精确不变量</a>；<a href="../runs/gemini_parity_novelty_review_20261010T0000Z/review.txt">严格新意评议</a>；<a href="../runs/mbrl_parity_coupling_v0/summary.json">首轮原生结果</a>；<a href="../runs/mbrl_parity_coupling_v1/summary.json">独立种子复核</a>。</p>')
composition_section=('<h2>组合一致性主张：边界结果保留，结构机制淘汰</h2><p>级联在 walker-run 留出任务上仍是条件性结果，但 cup-catch 与 walker-run-backwards 未复现。官方 TD-MPC2 的两步潜变量 commutator 也没有选择性升高：walker-run 为 cascade 0.762、random 0.862；cup-catch 为0.497、0.505；walker-walk 为0.817、0.914。严格评议给出新意3/10，认为“组合一致性损失”把普通多步暴露／误差累积、目标条件输入泄漏和未证实的非交换结构混成宽泛标签；因此不能把 walker-run 结果解释为反事实组合算子攻击。</p><p><a href="../configs/world_model_composition_integrity_screen.json">筛选结论</a>；<a href="../runs/tdmpc_commutator_diagnostic_v0/status.json">commutator 诊断</a>；<a href="../runs/gemini_compositional_integrity_review_20261010T0000Z/review.txt">严格机制评议</a>。</p>')
frequency_semigroup_section=('<h2>频率容量与半群缺陷：两种高新意候选均未通过</h2><p>为区分“合法数据造成共享表示频率偏置”和“合法静止经验造成多步半群缺陷”，在官方 TD-MPC2 上对 walker-run 进行了两种两种子短门槛。低频源轨迹相对随机没有降低 MPC 回报，反而在两个种子上略高；一步误差比随机为1.019倍，目标 action-Jacobian 条件数比为1.017倍。静止源轨迹同样提高 MPC 回报，三步组合缺陷相对随机仅0.995倍，一步误差为0.982倍，actor 影响还跨种子反向。因此不把频率错配或半群缺陷写成主张，也不进行长 GPU 扩展。</p><p><a href="../configs/world_model_frequency_semigroup_screen.json">筛选结论</a>；<a href="../runs/tdmpc_frequency_reservoir_gate_v0/status.json">频率门槛</a>；<a href="../runs/tdmpc_semigroup_defect_gate_v0/status.json">半群门槛</a>。</p>')
a2world_section=('<h2>Round41：A2World 序列级动作非局部性（已降级为架构观察）</h2><p>固定噪声下直接加载发布的 A2World 2B 去噪器：保持动作第1--4步不变、反转第5--20步，早期潜变量槽平均绝对差0.1779，后期0.1755；只翻转第20步时，早期0.0701，后期0.0626。该结果说明全局动作条件与双向注意力会把后缀变化传播到早期输出，但没有解码视频、任务损害或修复收益。Gemini 严格复核给新意2/10，认为这是普通非因果条件行为；下一道门只有闭环安全决策短测，并要求严格因果基线。</p><p><a href="../runs/a2world_fixed_noise_suffix_v0/summary.json">固定噪声结果</a>；<a href="../runs/a2world_global_action_condition_v0/summary.json">条件载体探针</a>；<a href="../runs/openwam_temporal_mask_audit_v0/summary.json">OpenWAM掩码审计</a>；<a href="../runs/gemini_round41_broadcast_temporal_nonlocality_20261010/review.json">Gemini复核</a>。</p>')
causal_role_section=('<h2>Round50--51：角色共享候选与开放前沿均未通过</h2><p>三种子 FACT 风格合成门控中，共享干路在失败样本只监督未来/价值后，失败批次未来误差比例为0.00687、价值误差比例为0.0917，但动作误差比例升至117.1、闭环终端距离比例为18.4；分离 outcome trunk 或停止 outcome 梯度后动作比例均为1.0。该现象只证明普通共享多任务梯度干扰可被构造。Gemini Round50 严格评议给3/10并拒绝，Round51 在排除投毒、OOD、遗忘、不确定性坍缩、视界/rollout、采样耦合、分支不一致、动作编码与 CoCo 后返回 REJECT_ALL。</p><p>近期动作编码不变性工作已经覆盖绝对/相对动作写法导致的等价轨迹崩溃，因此不把它重新包装成安全漏洞。当前硬约束下没有达到7/10且有公开载体的主张；不启动 FACT 长权重下载或 GPU 扩展。若继续，必须先明确放宽一个威胁模型约束，或把目标改为负边界/评测基准。</p><p><a href="../runs/causal_role_privilege_toy_v0/summary.json">合成门控</a>；<a href="../runs/gemini_round50_causal_role_strict_20261010/review.json">Round50严格复核</a>；<a href="../runs/gemini_round51_open_frontier_search_20261010/review.json">Round51开放检索</a>；<a href="https://arxiv.org/abs/2609.23252">动作编码不变性工作</a>。</p>')
latest_search_section=('<h2>最新候选搜索：保持停止条件</h2><p>严格搜索继续覆盖了 SimNorm 球面秩塌缩、WorldAgen 非交换动作块 TTT 漂移、PETS Fisher 秩匮乏、半群 Cauchy 缺陷和逆周期自确认等方向。当前结果中，SimNorm 与不确定性方向已有失败短测，逆周期方向被评为因果混淆／伪标签自训练的重包装；半群候选需要现有仓库没有的变量步长载体，WorldAgen Lie-bracket 候选缺少可直接调用的发布 LIBERO TTT trainer。Round6 又在同样的现有载体和排除条件下得到“没有候选越过门槛”的封闭性判断，因此本轮没有把“Gemini 给出的新意分数”当作证据，也没有启动长训练。Round6 首次请求曾超时，重试成功；超时请求没有被当作实验结果。</p><p><a href="../runs/gemini_mechanism_search_round4_20261010T0000Z/review.txt">Round4 搜索</a>；<a href="../runs/gemini_mechanism_search_round5_20261010T0000Z/review.txt">Round5 搜索</a>；<a href="../runs/gemini_mechanism_search_round5b_20261010T0000Z/review.txt">Round5b 搜索</a>；<a href="../runs/gemini_inverse_cycle_candidate_review_20261010T0000Z/review.txt">逆周期评议</a>；<a href="../runs/gemini_mechanism_search_round6_20261010T0000Z_retry/review.txt">Round6 严格封闭性评议</a>。</p>')
latest_search_section=('<h2>最新候选搜索：保持停止条件</h2><p>严格搜索继续覆盖了 SimNorm 球面秩塌缩、WorldAgen 非交换动作块 TTT 漂移、PETS Fisher 秩匮乏、半群 Cauchy 缺陷和逆周期自确认等方向。当前结果中，SimNorm 与不确定性方向已有失败短测，逆周期方向被评为因果混淆／伪标签自训练的重包装；半群候选需要现有仓库没有的变量步长载体，WorldAgen Lie-bracket 候选缺少可直接调用的发布 LIBERO TTT trainer。Round6 又在同样的现有载体和排除条件下得到“没有候选越过门槛”的封闭性判断。Round7 的最高分“分类 reward 支持在长 horizon 上变平”在原生 horizon=3 的两种子短测中失败：支持熵相对 random 为0.976倍，MPC 下降只有一枚种子达到17.2%。Bellman 自举自标签方向被评为4/10的普通 bootstrap／多任务干扰重包装。Round8 的措辞收敛重试返回 REJECT_ALL：在当前约束下没有新意达到7.0且可直接运行的候选。对 WorldAgen 原生 TTT 的“跨头预测—策略纠缠”候选也只得到4/10，属于普通共享 Transformer 多任务干扰，因此未运行训练。反事实查询侧信道在三个原生任务的几何搜索中也没有出现执行不可分而查询可分的配对。Round9 放宽到真实多客户端原生 TTT 后仍返回 REJECT_ALL，最高新意5.2；剩余方向都可归入普通负迁移、分布偏移／不可辨识性或 TTT 梯度偏置，未形成独立安全机制。Round10 的 prototype-router、counterfactual-branch 和 action-operator 三个搜索也分别被评为3、4和4分；最后一项的原生配对诊断没有显示 cascade 特异算子偏差。Round11 的“跨反事实分支耦合”进一步被评为4.5：现有 TD-MPC2 与 PETS 载体按点或独立边缘建模，没有跨分支联合头；在逐分支边缘保持时无法单独改变联合耦合，因此没有可运行的安全机制载体。Round13 对当前有序交互印刻主张评为5.5：时间反序和 actor 对照支持条件性现象，但多时域 latent 漂移与更新算子非交换性没有选择性，仍可能是非 IID 适配与规划误差累积。Round14 的 C1 真实 LIBERO 短筛选没有通过谱熵和完成率门槛；Round15 的 gradient-nullspace inversion 与 Dreamer RSSM contraction 分别重叠已否决的 gradient-integrity/mode-collapse 家族或依赖不可用载体。Round16 的 commutator 防御虽把局部交换子压低，却没有恢复 cascade seed31；Jacobian spectrum 防御在 kappa=5 未触发，kappa=3 又改变 clean/order controls，两个防御均未通过。Round17 的 TD-MPC2 组件搜索返回 REJECT_ALL，Round18 的预测权威倒置复核仅为4.5/10。Round19 将 actor-restore 作为硬约束复核，返回 REJECT_ALL、评分2.1/10，认为跨 seed 变号与默认 MPC 采样方差相容；Round20 的 provenance projection 得分7.3但 native gate 失败，Round21 的 trajectory-manifold/support 重写复核为4.0/10。Round22 CLCI/PNDR 虽获7.5/10 idea 分数，但 native gate 的 source-target action 子空间对齐为0.489（random0.769），三步 branch dispersion 基本不变，故拒绝。Round23 对“子空间排斥”重命名复核为4.0/10，确认不能再把几何差异升级为机制。单步训练窗口仍保留较小 cascade 损失，但 Q-only、hopper 顺序和 source seed22 控制未形成可升级机制。建议的五 seed 检验只是终止确认，不启动扩展。严格新意尚未达到7–7.5，因此本轮没有把“Gemini 给出的新意分数”当作证据，也没有启动长训练。Round6 首次请求曾超时，Round8 首次措辞曾被 provider content filter 拦截；这些失败请求均没有被当作实验结果。</p><p><a href="../runs/gemini_mechanism_search_round4_20261010T0000Z/review.txt">Round4 搜索</a>；<a href="../runs/gemini_mechanism_search_round5_20261010T0000Z/review.txt">Round5 搜索</a>；<a href="../runs/gemini_mechanism_search_round5b_20261010T0000Z/review.txt">Round5b 搜索</a>；<a href="../runs/gemini_inverse_cycle_candidate_review_20261010T0000Z/review.txt">逆周期评议</a>；<a href="../runs/gemini_mechanism_search_round6_20261010T0000Z_retry/review.txt">Round6 评议</a>；<a href="../runs/gemini_mechanism_search_round7_20261010T0000Z/review.txt">Round7 评议</a>；<a href="../runs/gemini_bellman_self_label_review_20261010T0000Z/review.txt">Bellman 评议</a>；<a href="../runs/gemini_mechanism_search_round8_retry_20261010T0000Z/review.txt">Round8 评议</a>；<a href="../runs/gemini_worldagen_ttt_review_20261010T0000Z/review.txt">WorldAgen TTT 评议</a>；<a href="../runs/gemini_mechanism_search_round9_20261010T0000Z/review.txt">Round9 评议</a>；<a href="../runs/gemini_routing_hijack_search_20261010T0000Z/review.json">Round10 路由评议</a>；<a href="../runs/gemini_new_carrier_search_20261010T0000Z/review.json">Round10 新载体评议</a>；<a href="../runs/gemini_action_operator_security_search_20261010T0000Z/review.json">Round10 算子评议</a>；<a href="../runs/gemini_counterfactual_coupling_search_20261010T0000Z/review.json">Round11 分支耦合评议</a>；<a href="../runs/gemini_claim_review_round13_20261010T0000Z/review.json">Round13 有序印刻复核</a>；<a href="../runs/gemini_mechanism_search_round14_20261010T0000Z/review.json">Round14 候选</a>；<a href="../runs/gemini_mechanism_search_round15_20261010T0000Z/review.json">Round15 严格搜索</a>；<a href="../runs/gemini_defense_search_round16_20261010T0000Z/review.json">Round16 防御候选复核</a>；<a href="../runs/gemini_mechanism_search_round17_20261010T0000Z/review.json">Round17 TD-MPC2 搜索</a>；<a href="../runs/gemini_mechanism_review_round18_20261010T0000Z/review.json">Round18 预测权威复核</a>；<a href="../runs/gemini_mechanism_search_round19_20261010T0000Z/review.json">Round19 actor-restore 严格复核</a>；<a href="../runs/gemini_mechanism_search_round20_20261010T0000Z/review.json">Round20 projection 复核</a>；<a href="../runs/gemini_mechanism_search_round21_20261010T0000Z/review.json">Round21 trajectory 复核</a>；<a href="../runs/gemini_mechanism_search_round22_20261010T0000Z/review.json">Round22 CLCI 复核</a>；<a href="../runs/tdmpc_clci_gate_v0/status.json">CLCI 原生短门</a>；<a href="../runs/gemini_mechanism_search_round23_20261010T0000Z/review.json">Round23 CLCI 排斥复核</a>。</p>')
counterfactual_topology_section=('<h2>Round72–73：反事实拓扑投毒评测候选</h2><p>Round72 的 TD-MPC2 反事实三角度量与 DreamerV3 对易子提案分别因载体路径错误、数学上恒为零和缺少可审计代码被拒绝。Round73 将主张收紧为：在采样式隐空间规划中，单步 factual 拟合保持正常时，未执行分支的价值排序可能被改变。Gemini 严格评议为 novelty 7.0，但结论是 ACCEPT_AS_EVALUATION，不是已成立主机制。</p><p>固定三步动作库的 TD-MPC2 短门在 walker-run seed32 出现 cascade 排序下降（Spearman 0.225→0.092，top-10 真实回报 0.843→0.820），seed31 未复制；walker-walk 只有 seed32 有较弱下降，hopper 的真实回报全为零。原生均值 Q、min-Q 和均值减标准差均未选择性修复排序。额外的 horizon=1 短门在 seed32 仅得到 Spearman 0.141/0.166/0.068（clean/random/cascade），top-10 真实回报 0.2690/0.2679/0.2682，未形成实际选择损害，因此候选保持条件性，不开长训练。Round74 扩到 walker-run 六个 seed 后，cascade-clean Spearman 平均差为 -0.061、top-10 真实回报平均差为 -0.0046，反序同样低、乱序反而较高，未形成稳定 source-order 效应。Round75 提出的 ActSWM 槽位置换和 TD-MPC2 连续任务 embedding 也在源码审计中被拒绝。Round76 检查的 Feedback World Model 官方仓库只有网页、图片、视频和 README，没有可执行模型与权重，因此不计入实验。Round77 的 World Action Planner 有扩散视频模型代码和环境，但 checkout 不含 checkpoint/Wan 基础权重，暂不下载大模型。Round78 的 MoSim Lie-generator/nilpotent-contraction 提案进一步被源码审计拒绝：实际只有 action-conditioned dynamics 与 odeint_adjoint/dopri5，没有 Lie、Magnus 或体积保持载体，因此未启动 GPU。</p><p><a href="../runs/gemini_round72_security_property_search_20261010/strict_review.json">Round72 复核</a>；<a href="../runs/gemini_round73_counterfactual_contract_claim_20261010/review.json">Round73 主张评议</a>；<a href="../runs/tdmpc_counterfactual_topology_gate_round73_v2/status.json">walker-run 排序</a>；<a href="../runs/tdmpc_branch_support_audit_round73/status.json">分支支持审计</a>；<a href="../runs/gemini_round78_native_candidate_search_20261010/source_audit.json">Round78 MoSim 源码审计</a>。</p>')
intervention_operator_section=('<h2>规划时干预算子候选：原生配对诊断未分离</h2><p>近期工作将“当前状态可读但动作后果传播错误”作为规划时干预保真度来测量。我们把它转成攻击假说：合法有序的 cheetah-run source curriculum 可能在共享 TD-MPC2 中旋转 walker-run 的动作干预算子，同时保持一步误差和当前状态捕获。复用已完成的 cascade/random dynamics-only 适配，在64个固定 walker-run 根状态上恢复同一物理动作对，并用冻结 latent-to-state readout 比较真实与想象的动作对比。</p><p>状态捕获三臂完全相同（18.4023；encoder 冻结）；干预余弦均值为 clean 0.7858、random 0.7603、cascade 0.7771，干预对比 MSE 为 clean 9.7611、random 10.8504、cascade 10.2094。cascade 没有相对 random 的特异算子偏差，不能解释 walker-run 的回报下降。严格评议也将该表述评为4/10，认为它重合于 intervention-fidelity 审计和普通跨任务负迁移，因此不升级、不扩展 GPU。</p><p><a href="../configs/world_model_intervention_operator_screen.json">筛选结论</a>；<a href="../runs/tdmpc_intervention_operator_v1/summary.json">原生配对诊断</a>；<a href="../runs/gemini_action_operator_security_search_20261010T0000Z/review.json">严格评议</a>。</p>')
temporal_order_section=('<h2>轨迹条件反事实完整性：最强候选仍是条件性现象</h2><p>在固定官方 TD-MPC2 mt30-48M、cheetah-run 300 步合法 source 与 dynamics-only 更新下，walker-run 两个留出种子的 cascade 回报为495.22/547.53，random 为612.08/678.78，保持相同动作多重集的时间乱序为613.76/669.45；一步误差比为0.922。把同一动作向量按时间反序后重新运行真实模拟器，回报恢复到608.43/621.95。actor 回报在所有 source 条件相同，因为 actor 参数没有更新。这组结果支持“有序交互而非动作预算或单纯 MPC 随机性”，但仍不足以证明独立内部原语。</p><p>作为强基线，80 次 upstream native full update 仍让 walker-run cascade 相对 random 下降19.0%/13.0%，reverse 仅下降6.0%/0.5%，shuffle 持平或上升，cascade 一步误差为 random 的0.993倍；但 full update 同时改变 actor，cup-catch seed32 也下降27.9%，所以 dynamics-only 结果仍是最干净的规划特异协议。 随后的 actor-restore 对照在每个条件结束时恢复官方 actor：cascade 相对 random 仅为−4.6%/+8.2%，reverse 为+0.2%/+17.1%，shuffle 为−2.3%/+13.1%，一步误差比仍为0.993。该结果削弱 planner-only 因果解释，full-update 结果只保留为联合更新的鲁棒性证据。</p><p>4×4 source-target 矩阵显示，cheetah-run、walker-walk 与部分 hopper source 能影响 walker-run，而 walker-walk/stand 目标大多保持不变；只有 cheetah-run source 同时通过两个种子的低一步误差门槛。固定候选动作库的模型/真实回报排序在 seed32 上 cascade Spearman 0.071、random 0.291、reverse 0.155、shuffle 0.354，seed31 则没有同样方向，因此排序诊断只保留为探索性支持。多时域 latent drift 与 A→B/B→A 更新差异也没有 cascade 特异性，拒绝把非正规或更新非交换性写成因果机制。Round14 的 C2 action-Jacobian desensitization 原生门槛同样失败（dither/random 回报比1.0、Jacobian Frobenius 比0.991、一步误差比1.016）。C1 TTT context-rank parasitism 评分7.4；真实 LIBERO 短筛选完成后，一步损失比为0.711，但隐藏表示谱熵只下降28.1%，六步 baseline 没有成功，且上游引用的 test_time_training/ttt_libero.py 缺失，因此未通过 C1 门槛。单步训练窗口控制仍保留 walker-run cascade 的12.4%/16.0%双 seed 损失，Q-only value 头、latent projection 短门、hopper 顺序控制和 cheetah source seed22 复测均未给出选择性新机制。Round20 provenance projection 得分7.3但 native gate 失败，Round21 trajectory-manifold/support 复核为4.0/10；严格复核给当前已测候选新意低于目标。</p><p><a href="../runs/tdmpc_order_controls_v0/status.json">时间顺序与 actor 对照</a>；<a href="../runs/tdmpc_full_update_order_controls_v0/status.json">full-update 强基线</a>；<a href="../runs/tdmpc_full_update_actor_restore_v0/status.json">actor-restore 因果强基线</a>；<a href="../runs/tdmpc_window_composition_h1_v0/status.json">单步训练窗口控制</a>；<a href="../runs/tdmpc_value_provenance_v0/status.json">Q-only value 控制</a>；<a href="../runs/gemini_mechanism_search_round21_20261010T0000Z/review.json">Round21 轨迹支持复核</a>；<a href="../runs/tdmpc_cascade_matrix_v0/status.json">source-target 矩阵</a>；<a href="../runs/tdmpc_planner_ranking_v3/status.json">规划排序诊断</a>；<a href="../runs/tdmpc_horizon_geometry_v0/status.json">多时域诊断</a>；<a href="../runs/tdmpc_update_noncommutativity_v0/status.json">更新算子诊断</a>；<a href="../runs/tdmpc_action_jacobian_desensitization_v0/status.json">C2 原生门槛</a>；<a href="../runs/worldagen_ttt_context_rank_v0/ttt20.json">C1 WorldAgen TTT 短筛选</a>；<a href="../runs/gemini_claim_review_round13_20261010T0000Z/review.json">严格主张复核</a>；<a href="../runs/gemini_mechanism_search_round14_20261010T0000Z/review.json">Round14 候选</a>。</p>')
topographic_section=('<h2>分类 reward 支持候选：原生 horizon 门槛失败</h2><p>严格搜索提出“合法近零回报边界轨迹让 TD-MPC2 的离散 reward 支持在长想象 horizon 上变平”。为保留原生协议，短测固定官方 horizon=3、未改 planner，只用 steady 与 random 合法源数据做40次 full update。steady 相对 random 的 MPC 回报在两个种子下降17.2%和8.9%，而固定验证状态上的分类 reward 熵比只有0.976，一步误差比0.998；actor 在第二种子还上升44.8。熵方向和两种子 planner 门槛均失败，长 horizon 版本需要非原生改动，因此停止。</p><p>文献边界也已收紧：近期 <a href="https://arxiv.org/abs/2606.18697">SWAAP</a> 已覆盖隐蔽数据投毒，<a href="https://arxiv.org/abs/2605.01950">TRAP</a> 已覆盖尾部轨迹排序后门；综述将奖励／安全模型投毒、trajectory ranking 与 predictive safety illusion 分开归类。因此这些方向不会被改名后当作本报告的新机制。</p><p><a href="../configs/world_model_topographic_support_screen.json">筛选设定与结论</a>；<a href="../runs/tdmpc_topographic_entropy_gate_v0/status.json">原生短测</a>；<a href="../runs/gemini_mechanism_search_round7_20261010T0000Z/review.txt">严格候选搜索</a>。</p>')
corrective_section=('<h2>规划纠正数据缺失：闭环持续性未建立</h2><p>为检验“合法源经验造成规划偏置后，受害者只收集自身 MPC 轨迹而缺少纠正支持”的闭环机制，固定 seed41 比较 clean、random-source、cascade-source，并在 handoff 后用20次 full update 续训。cascade handoff 的200步回报为233.91，高于 random-source 的181.26；MPC-only 续训为228.14，混入随机真实轨迹后反而为148.21。三臂动作协方差秩都为6，没有支持缺失或随机数据恢复的可分离信号，因此不把该分支写成持续性攻击。</p><p><a href="../configs/world_model_corrective_starvation_screen.json">设定与停止条件</a>；<a href="../runs/tdmpc_corrective_starvation_gate_v0/status.json">原生闭环短测</a>；<a href="../scripts/pilot_tdmpc_corrective_starvation.py">运行脚本</a>。</p>')
cem_covariance_section=('<h2>CEM 协方差收缩候选：性能损害未与覆盖效应分离</h2><p>Round7 提出“合法边界动作数据注入极端协方差，使原生 CEM 更快收缩到错误动作分布”。在 MBRL-Lib CartPole 上固定3000条真实转移、同一训练种子和原生 CEM（100 population、3 iterations、horizon15），比较 random、boundary 与等幅 interior sham，并用810—813四个评估种子记录每轮全 population 协方差迹。boundary 的平均回报为136.5，低于 interior 的183.5和 random 的196.25；但第2轮/第0轮协方差比仅为0.57007，对比 interior 0.57559、random 0.57894，差异很小，且 boundary 的初始协方差更高。因而回报损害没有显示出边界特异的 CEM pinching 机制，归因仍可由覆盖差异和模型误差解释，不升级或扩展。</p><p><a href="../configs/world_model_cem_covariance_pinching_screen.json">筛选结论与完整数字</a>；<a href="../scripts/pilot_mbrl_rank_general.py">训练与 CEM 记录脚本</a>；<a href="../runs/mbrl_covariance_pinching_eval_boundary_metrics_v1/summary.json">boundary 复核</a>。</p>')
contractive_basin_section=('<h2>MPPI 伪吸引子候选：周期轨迹未造成安全损害</h2><p>Gemini 将“合法周期动作让 MPPI 候选分布提前收缩到伪吸引子”评为新意7.3的短测候选。官方 TD-MPC2 中，cheetah-run 周期轨迹在 walker-run 上的 planner 方差比随机略低（0.297 对0.318），但回报反而提高到402.08；一步误差是随机的2.37倍，违反低误差不变量。reacher-easy 上周期轨迹既没有损害，也没有超过等RMS对照的方差收缩。因此按门槛淘汰，不扩大训练，不实现熵下限防御。</p><p>详见<a href="../configs/world_model_contractive_basin_screen.json">筛选结论</a>、<a href="../runs/tdmpc_contractive_basin_gate_v0/status.json">walker-run gate</a>、<a href="../runs/tdmpc_contractive_basin_gate_reacher_v0/status.json">reacher-easy gate</a>、<a href="../runs/gemini_next_mechanism_search_20261010T0000Z/assessment.json">Gemini 候选评审</a>。</p>')
horizon_shadowing_section=('<h2>时域同步 bootstrap 候选：跨种子方向反转</h2><p>随后测试了 Gemini 提出的“周期5轨迹与 TD-MPC2 rollout horizon 同步，干扰 terminal value bootstrap”的候选。period-5 在 walker-run seed11 比随机低约3%，一步误差反而更低；但 seed12 比随机高约14.2%，300 次更新剂量也没有恢复稳定损害，MPPI 方差没有出现特异收缩，等RMS dither 在 seed12 和 dose-300 中还更低。因此没有 horizon-specific 机制或稳定任务损害，停止扩展。</p><p>详见<a href="../configs/world_model_horizon_shadowing_screen.json">筛选结论</a>、<a href="../runs/tdmpc_horizon_shadowing_gate_period5_v0/status.json">seed11</a>、<a href="../runs/tdmpc_horizon_shadowing_gate_period5_seed12_v0/status.json">seed12</a>、<a href="../runs/tdmpc_horizon_shadowing_gate_period5_dose300_v0/status.json">dose-300</a>。</p>')
origin_section=('<h2>现实与想象来源识别：固定确认未通过</h2><p>'+html.escape(origin_security['report_note_cn'])+
    '</p><p>两个独立回合共1000个环境决策步；累计9,223,470个预测转移，冻结模型保持不变。'+
    '未进行世界模型训练；研究实验无API调用，备用接口连通性另测一次生成。'+
    '首轮线性／PCA探针未通过；一次错误维度的RBF试验已排除，CPU成本保留。'+
    '更正后的RBF在开发集通过后固定参数与阈值，再评估两个新回合。'+
    '详见<a href="../configs/world_model_origin_security.json">研究结论</a>、'+
    '<a href="../'+html.escape(origin_security['source'])+'">设计、先行工作与成本</a>、'+
    '<a href="../'+html.escape(origin_security['raw_confirmation'])+'">原始确认结果</a>。</p>')
recurrence_section=('<h2>递归动力学安全短测：自激反馈未建立</h2><p>'+html.escape(recurrence_security['report_note_cn'])+
    '</p><p>详见<a href="../'+html.escape(recurrence_security['source'])+'">操作、因果对照与成本</a>、'+
    '<a href="../runs/tdmpc_recurrence_feedback_v0/summary.json">固定注入回放对照</a>。'+
    '当前实验接口按用户要求使用官方 Gemini；接口成功不计入机制证据。</p>')
geometry_section=('<h2>潜空间坐标变换：等价于目标距离改变</h2><p>'+html.escape(geometry_security['report_note_cn'])+
    '</p><p>详见<a href="../configs/world_model_geometry_security.json">候选边界</a>、'+
    '<a href="../runs/latent_gauge_equivalence_v0/summary.json">CPU数值对照</a>。'+
    '本轮Gemini评议约7.3秒成功，模型意见不作为新意或实验结论。</p>')
goal_section=(
    '<h2>主线候选：共享动作 token 因果总线</h2>'
    '<p>指令条件世界模型可能把 goal 写入 contextualized input-action token，使直接 goal→world 掩码仍允许目标改变固定状态与固定动作下的物理预测。WorldAgen 扩展三十窗口复现实验覆盖公开 LIBERO-10 数据集的十个 episode，原生 goal swap 的 world relative L2 平均为 0.300，空指令为 0.270；切断九条原生 goal→input-action 边后，world 输出在所有窗口精确不变，policy 输出仍保持 goal sensitivity（relative L2 平均 0.176）。记录目标、换目标、edge-cut 的 primary 下一帧 patch MSE 为 1.423、1.496、1.484，显示约 4.3% 的预测代价；30 个窗口中换目标 MSE 全部上升，bootstrap 95% 区间为 world relative L2 0.244–0.359、MSE 差 0.0465–0.1030，edge-cut policy relative L2 0.139–0.217。</p>'
    '<p>独立的 AlphaBrain CosmosPolicy 载体在三组固定 latent 任务嵌入交换中，action、future proprio、future wrist、future primary 和 value 槽均发生非零变化；在第一组任务对加入 row-wise context/self-attention cut 后，future proprio/wrist/primary 三个 world 槽的差异精确为 0，而 action/value 槽仍保持平均 0.00692 的 goal 敏感性。随后在七个真实 LIBERO 图像/state 窗口上、固定各窗口记录的16步执行动作块重复同一门：native future-world 差异平均 0.03105、每个窗口非零，edge-cut 后精确归零，action/value 仍平均变化 0.01629。该第二载体仍没有环境步或任务成功率，因此当前结论是“跨载体固定 latent 与真实输入的攻击—修复短门通过，闭环任务待验证”，而非已完成泛化。候选自评 novelty 7/10。</p>'
    '<table><thead><tr><th>载体与证据</th><th>窗口／重复行</th><th>native world</th><th>edge-cut world</th><th>policy after cut</th><th>forecast cost</th></tr></thead><tbody>'
    '<tr><td>WorldAgen real-input goal swap</td><td>30</td><td>rel. L2 0.2999</td><td>0</td><td>rel. L2 0.1760</td><td>MSE +0.0732</td></tr>'
    '<tr><td>AlphaBrain real-input one-step</td><td>7</td><td>0.03105</td><td>0</td><td>0.01629</td><td>held-out target not run</td></tr>'
    '<tr><td>AlphaBrain sampler offsets 1/5/9</td><td>3 / 9 rows</td><td>0.07452</td><td>0</td><td>native policy branch</td><td>held-out target not run</td></tr>'
    '<tr><td>AlphaBrain expanded held-out forecast</td><td>21 / 7 episodes</td><td>goal-world 0.05317</td><td>fixed-action 21/21 zero</td><td>native policy branch</td><td>repair +0.04899, episode CI [0.03321, 0.06798]</td></tr>'
    '</tbody></table><p><a href="../runs/goal_bus_main_results_v0/summary.json">机器可读主结果</a>；重复 offset 的区间按物理 episode 聚类，zero-context MSE 0.60833 只作 prompt-independent 诊断。旧 full edge-cut 是过强 legacy comparator。</p>'
    '<p>'+html.escape(goal_bus['working_claim_cn'])+'</p>'
    '<p>全部合法 offset 扫描：同一 30 个物理窗口的 9 个任务 offset 共 270 个重复配对行；每个 offset 的 native world 差异均非零，edge-cut world 差异均为 0，policy 均敏感。按物理窗口聚类的 bootstrap 95% 区间为 world relative L2 0.286–0.361、edge-cut policy 0.163–0.240、换目标减记录目标 MSE 0.0626–0.0940。详见<a href="../runs/worldagen_goal_bus_offset_sweep_v1/summary.json">offset 扫描</a>、<a href="../runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json">聚类统计</a>与<a href="figures/worldagen_goal_bus_offset_sweep.svg">SVG 图</a>。</p>'
    '<p>远距合法 goal 对照：任务表中相隔 5 位的替换指令在 30 个真实窗口产生 native world relative L2 平均 0.333，edge-cut 后精确归零，policy relative L2 仍为 0.181；primary MSE 差的 bootstrap 95% 区间为 0.0622–0.1012，29/30 为正。详见<a href="../runs/worldagen_goal_bus_far_goal_v3/summary.json">远距 goal 结果</a>与<a href="../runs/worldagen_goal_bus_far_goal_v3/statistics.json">配对统计</a>。</p>'
    '<p>任务层短基线：WorldAgen 在 LIBERO spatial/object/goal 三个任务族各取一个固定初始状态，native、edge-cut、empty-goal 共 9 回合，每回合 36 步均未成功；edge-cut 与 native 动作范数接近。该时域远短于官方 600 步，只作为控制保留 sanity check，不作成功率结论。详见<a href="../runs/worldagen_libero_closed_loop_v0/summary.json">短闭环结果</a>与<a href="../configs/world_model_goal_bus_security.json">主线设定</a>。</p>'
    '<p>官方 600 步时域的控制 pilot 也已完成：spatial/object/goal 各一个固定初始状态，native 与 edge-cut 共 6 回合均为 0/3 成功，动作范数均值约 1.780 与 1.794。由于释放 checkpoint/configuration 没有通过 clean task gate，这不能支持任务损害归因；它只作为控制保留负对照。详见<a href="../runs/worldagen_libero_closed_loop_full_pilot_v0/combined_summary.json">600 步控制 pilot</a>。</p>'
    '<p>证据见<a href="../runs/worldagen_goal_bus_replicate_v2/summary.json">WorldAgen 三十窗口实验</a>、<a href="../data/worldagen_goal_probe_extra/provenance.json">扩展数据 provenance</a>、<a href="../runs/worldagen_goal_bus_replicate_v2/statistics.json">三十窗口配对统计</a>、<a href="../runs/worldagen_goal_invariance_v0/summary.json">冻结模型结果</a>、<a href="../runs/worldagen_goal_paths_v0/summary.json">原生掩码通路</a>、<a href="../runs/alphabrain_cosmos_policy_goal_bus_audit_v0/summary.json">AlphaBrain 载体审计</a>、<a href="../runs/alphabrain_cosmos_policy_goal_swap_v0/summary.json">AlphaBrain 固定 latent 攻击—修复短门</a>、<a href="../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/summary.json">AlphaBrain 七窗口真实输入短门</a>、<a href="../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/vae_provenance.json">公开 VAE provenance</a>；主张与下一门槛见<a href="../configs/world_model_goal_bus_security.json">设定文件</a>。</p>'
)

goal_section += '<p>新增第一组任务对通路消融：仅切断观察槽的直接文本 cross-attention 时，future proprio/wrist/primary 的平均绝对差异仍为 0.00448；进一步阻断它们读取 action/value 槽后差异才归零。零 T5 context 诊断下，原生 world 槽平均差异为 0.01535，修复后为 0，action/value 平均差异仍为 0.01403。零 context 不是已验证的训练分布条件。AlphaBrain 当前输入是随机 latent，动作槽也参与联合预测，尚未固定真实已执行动作；因此这里只支持间接文本通路与局部隔离结果，不把它算作主张中固定物理动作不变量的完整跨载体复现。随后用备用环境的原生 transformer-engine 运行时重放同一门：三组交换仍改变全部预测槽，第一组修复后的 future world 差异精确为 0，action/value 平均差异为 0.00692；这仍是随机 latent、联合动作预测，不能替代固定真实已执行动作。</p>'
goal_section += '<p>真实输入的因果阶梯进一步确认中间载体：只切断 world rows 的直接文本 cross-attention，三个窗口的 world 差异仍非零，均值0.04235；同时清除 action/value carrier 的文本后才得到 fixed-action world cut 的精确零差异。详见<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_refined_cut_v0/summary.json">world-text-only 消融</a>和<a href="../scripts/analyze_alphabrain_refined_cut.py">聚合脚本</a>。</p>'
goal_section += '<p>第二载体真实输入门已通过七个不同 episode：公开 Wan2.1 VAE 将真实 LIBERO 图像/state 编码为 [1,16,9,28,28]，native future-world 差异平均 0.03105，覆盖五种任务对，bootstrap 95% CI 为 0.02896–0.03339，edge-cut 后精确归零，action/value 仍保持平均 0.01629（bootstrap 95% CI 0.01423–0.01865）的 goal 敏感性。零 T5 context 诊断的 native world 差异平均 0.05517，cut 后仍为 0；它不是发布模型的 no-goal 训练条件。该门使用公开 VAE 路径而非 gated NVIDIA tokenizer 文件，且只做单步冻结 DiT forward；没有环境步或任务成功率结论。详见<a href="../scripts/pilot_alphabrain_cosmos_policy_real_input_goal_bus.py">真实输入门脚本</a>、<a href="../scripts/analyze_alphabrain_real_input_goal_bus.py">聚合脚本</a>与<a href="../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/summary.json">七窗口结果</a>。</p>'
goal_section += '<p>随后用官方 CosmosPolicy 两步 reverse-ODE sampler 做固定动作确认：在 episode 3–9 的七个真实窗口中，同时固定 observation 槽与记录的16步 action 槽，native future-world goal-swap 差异均非零，平均0.05589，bootstrap 95% CI为0.03660–0.07801；matched action-bus cut 后七窗口 world 差异精确为0，action/value policy 差异平均0.06373（bootstrap 95% CI为0.02571–0.10887）。该结果是 sampler confirmation，没有环境步或任务成功率结论。详见<a href="../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_goal_bus.py">sampler 门脚本</a>、<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0/summary.json">七窗口 sampler 结果</a>。</p>'
goal_section += '<p>又在三个物理窗口上补做合法 task offset 1、5、9 的跨语义距离矩阵，共9个重复配对行。每行 native world 差异均非零，均值为0.07452，按 episode 聚类的 bootstrap 95% CI为0.02845–0.10800；三个 offset 的 edge-cut world 差异全部精确为0，policy 仍保持敏感。重复 offset 按物理 episode 聚类，不把9行当作9个独立世界。详见<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_offsets_v0/summary.json">offset sampler 聚合</a>和<a href="../scripts/analyze_alphabrain_sampler_offsets.py">聚合脚本</a>。</p>'
goal_section += '<p>为了把“修复代价”从槽位差异推进到预测指标，又把七个 episode 的三个时间点共21个窗口当前时刻之后16帧的真实 LIBERO 图像/state 用同一公开 VAE 编码为 held-out latent target，target 没有输入 denoiser。21窗口 native goal-world 差异均非零，均值0.05317（episode-cluster bootstrap 95% CI 0.04512–0.06180），fixed-action world cut 后21/21精确为0；held-out repair MSE delta 平均+0.04899（episode-cluster CI 0.03321–0.06798，7/7 episode 聚类均值为正）。该指标仍受 sampler noise 影响，seed-2 三窗口 repair delta 为−0.00079，因此不把正代价写成普遍规律。作为 prompt-independent 诊断，zero T5 context 的 MSE 是0.60833；它没有经过 goal-free dynamics 训练，不能当作强基线。详见<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json">扩展21窗口 matrix</a>、<a href="../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix.py">matrix脚本</a>和<a href="../scripts/analyze_alphabrain_forecast_matrix.py">聚合脚本</a>。</p>'
goal_section += '<p>独立 sampler noise seed 2 的三窗口复现中，native world 差异仍3/3非零，fixed-action cut 仍3/3精确为0，均值0.06270；但 held-out MSE repair delta 为−0.00079。因此 exact invariance 与 action sensitivity 是稳定主证据，+0.06737 forecast cost 仅作为 seed-1 的条件性 fidelity 结果。详见<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json">noise-seed 复现</a>。</p>'
goal_section += '<p>repair 的物理动作对照：在三个窗口中只替换为同一 episode 后续记录的另一段动作，fixed-action world cut 后 action→world 差异均非零，均值0.08715。该结果说明 corrected cut 保留了动作条件；policy 输出从并行 native branch 读取。详见<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_action_control_v0/summary.json">动作 control</a>与<a href="../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_cut_v0/summary.json">fixed-action cut 聚合</a>。</p>'
goal_section += '<p>逐窗口 forecast 曲线见<a href="figures/alphabrain_forecast_tradeoff.svg">forecast tradeoff SVG</a>；区间估计以七个 episode cluster 为单位，而不是把 21 个重复时间点或 latent 坐标当作独立样本。</p>'
goal_section += '<p>随后将同一官方 sampler 接入 LIBERO 环境，分别在 spatial、object、goal 三个任务族各跑一个 task0 初始状态，native 与 corrected fixed-action world cut 各执行32步。两种条件均为0/3成功；动作范数均值分别为0.415和0.653。该结果是修正后控制保留的短时域 sanity check，不能支持任务损害归因，也不能替代官方长时域 clean baseline。详见<a href="../runs/alphabrain_cosmos_libero_short_control_fixed_action_v1/aggregate_summary.json">修正 AlphaBrain LIBERO 短闭环聚合</a>、<a href="../scripts/pilot_alphabrain_cosmos_libero_short_control.py">运行脚本</a>和<a href="../scripts/analyze_alphabrain_libero_short_control_fixed.py">聚合脚本</a>。为确认失败不是短 horizon 偶然现象，spatial task0 又运行到608步，native与fixed-action cut均为0/1；详见<a href="../runs/alphabrain_cosmos_libero_full_control_v1/aggregate_summary.json">608步控制 pilot</a>。</p>'
uncertainty_section=('<h2>不确定性分解短测：分布保持前提未通过</h2><p>'+html.escape(uncertainty_screen['report_note_cn'])+
    '</p><p><a href="../runs/uncertainty_decomposition_v0/summary.json">原始数值结果</a>；'+
    '<a href="../configs/world_model_uncertainty_screen.json">候选边界与后续要求</a>。</p>')
rejected_section=('<h2>被淘汰预测的影响：正常选择能力未达门槛</h2><p>'+html.escape(rejected_future['report_note_cn'])+
    '</p><p><a href="../runs/rejected_future_influence_v1/summary.json">短测结果</a>；'+
    '<a href="../configs/world_model_rejected_future_screen.json">适用边界与后续要求</a>。'+
    '方法入口：<a href="https://arxiv.org/html/2607.02403v1#S3.SS3">ACID 自适应权重</a>；'+
    '相关先例：<a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002607">归一化与劣势选项效应</a>。</p>')
reliance_rows=''.join('<tr><td>'+html.escape(r['task'])+'</td><td>'+format(r['native_return'],'.2f')+
    '</td><td>'+format(r['action_blind_return'],'.2f')+'</td><td>'+format(r['native_prior_return'],'.2f')+'</td></tr>'
    for r in dynamics_reliance['effects'] if r['clean_gate_passed'])
reliance_section=('<h2>原生任务基线：动作条件动力学的实际作用</h2><p>'+html.escape(dynamics_reliance['report_note_cn'])+
    '</p><table><thead><tr><th>任务</th><th>完整规划</th><th>动力学忽略动作</th><th>仅策略网络</th></tr></thead><tbody>'+
    reliance_rows+'</tbody></table><p><a href="../runs/tdmpc_dynamics_reliance_v0/summary.json">完整回合结果</a>；'+
    '<a href="../configs/tdmpc_dynamics_reliance_screen.json">操作范围与后续任务选择</a>。</p>')
symmetry_section=('<h2>策略分布保持候选：数值构造未通过</h2><p>'+html.escape(policy_symmetry['report_note_cn'])+
    '</p><p><a href="../runs/tdmpc_policy_symmetry_v0/summary.json">中止前的原生回合</a>；'+
    '<a href="../runs/tdmpc_policy_symmetry_v1/summary.json">三任务数值检验</a>；'+
    '<a href="../configs/world_model_policy_symmetry_screen.json">结论范围</a>。</p>')
document=document.replace('<h2>已完成的实现</h2>',symmetry_section+reliance_section+rejected_section+uncertainty_section+goal_section+geometry_section+recurrence_section+origin_section+sampling_section+wav_alias_section+boundary_stiffness_section+replay_priority_section+timeout_laundering_section+non_normal_section+temporal_order_section+parity_section+composition_section+frequency_semigroup_section+a2world_section+causal_role_section+latest_search_section+counterfactual_topology_section+intervention_operator_section+topographic_section+corrective_section+cem_covariance_section+contractive_basin_section+horizon_shadowing_section+experiment_section+baba_section+drwell_section+revision_section+lazy_section+acquisition_section+reactor_grammar_section+'<h2>已完成的实现</h2>',1)
document=document.replace('<h1>真实经验能否劫持集体学习？</h1>',
                          '<h1>世界模型安全：机制探索与原生任务证据</h1>',1)
status_start=document.index('<div class="status">')
status_end=document.index('</div>',status_start)+len('</div>')
document=(document[:status_start]+'<div class="status"><strong>当前主线：共享动作 token 因果总线（第二载体固定 latent 与真实输入攻击—修复短门通过，闭环待验证）。</strong> '+
    html.escape(policy_symmetry['report_note_cn'])+
    '此前9回合原生基线确认：取消推演中的动作条件，行走975.51→80.22、接球994→718，到点仍为946；全部模型权重不变。仅策略网络在接球/到点同样高分，在行走仅500.25。后续据此选择任务和强基线；这不是新的攻击证据，仅覆盖每任务一个种子。'+
    '此前候选“被淘汰预测也能影响选择”的组合控制器仅完成62/128个正常目标，未通过80%门槛，未执行攻击；不能据此宣称原生WAV或ACID失效。归一化的劣势选项效应已有先例。'+
    '此前不确定性重分配短测未保持原生预测分布，暂不推进该构造；这不是对所有不确定性攻击的否定，也不要求后续新想法必须保持预测完全不变。'+
    '最新发布权重短测确认：直接掩码没有阻断目标经动作 token 进入世界预测；固定物理输入换目标的差异，在切断 9 条连接后归零。'+
    '这是带有输入兼容性限制的预测层因果结果，未建立任务损害或区别于已有指令泄漏研究的主贡献。'+
    '此前坐标变换短测归结为目标距离改变，未开展该方向的模型攻击。'+
    '此前单秩动力学修改保留一步预测，但目标动作门槛未通过；切断注入反馈的对照复现或超过其效果，自激机制未建立，未运行任务攻击。'+
    '此前来源识别短测的正常控制得994.12和974.09分，但其中一个独立回合误报1.61%，超过预设1%门槛。'+
    '早期安全方向已有预测损害与恢复障碍，但稳定任务损失未建立。'+
    (html.escape(epistemic_collapse['report_note_cn'])+' 成员分歧压平的一步平衡均值验证通过，但均值规划也退化，且每个种子未达到预设20%任务损失门槛；不把它升级为风险专属攻击。' if epistemic_collapse_path.exists() else '')+
    (html.escape(conductivity_gemini['report_note_cn'])+' 作为执行器复核，攻击经验造成的错误灯泡预测仍能被 Gemini 执行器通过真实查看纠正；三臂均成功。' if conductivity_gemini_path.exists() else '')+
    html.escape(sampling_security['headline_cn'])+
    ' 后续采样梯度构造的干净控制门槛也未通过（364.32/1000，要求900），且尚无区别于既有梯度操纵的核心贡献；未运行该构造的攻击组。'+
    ' 最新三十窗口复现实验在固定图像、状态和动作下，goal swap 的 world relative L2 平均为0.300，空指令为0.270；切断九条 goal→input-action 边后所有 world 输出精确不变，policy 仍保持 goal-sensitive，记录目标／换目标／修复 primary forecast MSE 为1.423／1.496／1.484。独立 AlphaBrain CosmosPolicy 三组固定 latent 任务交换和七个真实图像/state 窗口均改变 future world 槽；matched row-wise cut 后 future proprio/wrist/primary 精确不变，action/value 仍保持 goal 敏感。该候选自评 novelty 7/10，下一门槛是多任务 forecast、no-goal/repair 对照和闭环验证。</div>'+document[status_end:])
document=document.replace(
    '目标仍为会议论文级证据：至少三类核心任务、强交互基线、多个可核验骨干、两种学习机制、独立 Agent 的真实任务损失，以及一组决定性的机制干预。原生 ALFWorld／ScienceWorld／DiscoveryWorld 任务作为外部验证。规模以短测效应和实际调用成本决定；目前所有确认性规模均为计划值。',
    '目标仍为ICLR会议论文级证据：形成一个广泛、清晰的世界模型安全主张，在多类核心任务上用强基线、多个可核验模型与决定性的机制干预验证安全影响。允许依据反证修改安全机制；不能把预测变化、某个实现缺陷、一般学习能力提升或一个任务中的基线改进当作最终贡献。只有短测支持最终机制后才扩大实验；目前新意7–7.5目标和确认性规模均未达成。',1)
document=document.replace('候选主张：一个 Agent 可以通过真实交互，把其他 Agent 的学习带入自我维持的错误循环。',
    '研究目标：提出新的世界模型安全机制，并在多个核心任务中用强基线与因果干预验证。',1)
(ROOT/'reports/research_progress.html').write_text(document)
print(json.dumps({'completed_result_rows':len(rows),'requests':sum(x['requests'] for x in usage.values()),
                  'report':'reports/research_progress.html'},ensure_ascii=False))
