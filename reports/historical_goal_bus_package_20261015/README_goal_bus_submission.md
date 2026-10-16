# Goal-bus paper submission package

This directory contains the current prediction–mechanism–decision paper package for:

> Instruction-conditioned world models can write task text into a shared action-token causal bus, changing a fixed-state/fixed-action physical forecast despite masked direct goal-to-world edges; when an independent planner consumes that forecast, the same intervention changes action ranking and real trajectories, while the utility direction remains planner-dependent.

The Chinese focused draft is [`goal_bus_paper_draft_cn.md`](goal_bus_paper_draft_cn.md). The integrated prediction–mechanism–decision report is [`goal_bus_iclr_research_report_cn.md`](goal_bus_iclr_research_report_cn.md). The English Markdown draft, LaTeX source, and compiled PDF contain the current decision extension. The [completion audit](goal_bus_completion_audit_cn.md) separates supported forecast-level and independent-planner conclusions from the still-open native task-level requirements.

## Evidence entry points

- Premise-first physical invariance: the causal estimand is (Y^{physical}\perp Goal\mid(S,A)) under fixed state and executed actions. The 120/120 same-scene replay and its boundary conditions are recorded in [`goal_bus_physical_invariance_spec.json`](../configs/goal_bus_physical_invariance_spec.json) and [`same-scene analysis`](../runs/worldagen_same_scene_goal_v1/analysis.json).

- WorldAgen 30-window aggregate: `../runs/worldagen_goal_bus_replicate_v2/summary.json`
- WorldAgen 270 legal-goal-pair aggregate: `../runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json`
- WorldAgen leave-one-offset-out stability: `../runs/worldagen_goal_bus_offset_stability_v0/summary.json`
- WorldAgen same-episode alternate-action control: `../runs/worldagen_goal_bus_action_control_v1/summary.json`
- WorldAgen matched policy-only edge-cut control: `../runs/worldagen_goal_bus_edge_cut_controls_v0/summary.json`
- WorldAgen six-task semantic grouping: `../runs/worldagen_goal_bus_task_groups_v0/summary.json`
- AlphaBrain 21-window matrix: `../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json`
- AlphaBrain seed-2 replicate: `../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json`
- AlphaBrain seed-3 replicate: `../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json`
- AlphaBrain task-stratified matrix: `../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_by_task_v0/aggregate.json`
- AlphaBrain matched goal-vs-action effect size: `../runs/alphabrain_goal_vs_action_effect_v0/summary.json`
- WorldAgen activation mediation matrix: `../runs/worldagen_goal_bus_activation_patch_v1/summary.json`
- Corrected fixed-score six-task action-ranking matrix (primary-view/first-chunk Forecast Error): `../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_v2_tasks6.json`
- Fixed-score six-task, three fresh-replay repeat check: `../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6_repeat3.json`
- Fixed-score six-task second-seed replicate: `../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6_seed23.json`
- Pooled task-seed replicate statistics: `../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_seed_replicate_statistics.json`
- Replicate aggregation code: `../scripts/compare_worldagen_counterfactual_replicates.py`
- Activation-patch decision repair ablation: `../runs/worldagen_counterfactual_action_ranking_v0/summary_activation_patch_tasks02.json`
- Activation-patch six-task decision repair ablation: `../runs/worldagen_counterfactual_action_ranking_v0/summary_activation_patch_tasks6_v2.json`
- Activation-patch six-task statistics (corrected primary-view/step-3 Forecast Error): `../runs/worldagen_counterfactual_action_ranking_v0/summary_activation_patch_tasks6_v2_statistics.json`
- Activation-patch ranking code: `../scripts/pilot_worldagen_activation_patch_action_ranking.py`
- Step-aligned calibrated planner summary: `../runs/worldagen_aligned_planner_v1/summary.json`
- Step-aligned calibrated planner statistics: `../runs/worldagen_aligned_planner_v1/statistics.json`
- Step-aligned planner code: `../scripts/pilot_worldagen_aligned_planner.py`
- Three-seed aligned planner replicate statistics (18 task-seed pairs): `../runs/worldagen_aligned_planner_replicates_v1/statistics.json`
- Aligned planner replicate aggregation code: `../scripts/aggregate_worldagen_aligned_replicates.py`
- Aligned planner replicate figure (SVG/PDF): `figures/worldagen_aligned_planner_replicates.svg`, `figures/worldagen_aligned_planner_replicates.pdf`
- Three-seed aligned planner, step-3 selection with 30-step real replay: `../runs/worldagen_aligned_planner_h30_replicates_v1/statistics.json`
- 30-step aligned consequence figure (SVG/PDF): `figures/worldagen_aligned_h30_replicates.svg`, `figures/worldagen_aligned_h30_replicates.pdf`
- Legal Goal offset dose-response sweep (task 0/2, offsets 1--9): `../runs/worldagen_aligned_goal_sweep_h30_v1/statistics_sweep.json`
- Legal Goal dose-response figure (SVG/PDF): `figures/worldagen_aligned_goal_sweep.svg`, `figures/worldagen_aligned_goal_sweep.pdf`
- Legal Goal sweep analysis/plot code: `../scripts/analyze_worldagen_aligned_goal_sweep.py`, `../scripts/plot_worldagen_aligned_goal_sweep.py`
- Three-seed legal Goal sweep aggregate (18 task-seed pairs, 30-step replay): `../runs/worldagen_aligned_goal_sweep_h30_replicates_v1/statistics.json`
- Three-seed sweep aggregation code: `../scripts/aggregate_worldagen_aligned_goal_sweep.py`
- Three-seed legal Goal dose-response figure (SVG/PDF): `figures/worldagen_aligned_goal_sweep_replicates.svg`, `figures/worldagen_aligned_goal_sweep_replicates.pdf`
- Six-task three-seed legal Goal sweep aggregate (18 task-seed pairs, 162 counterfactuals, 30-step replay): `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/statistics.json`
- Forecast-error mediation diagnostic over the 162 counterfactuals: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/mediation_statistics.json`
- Fixed Clean-anchor / Bus-Cut decision-repair path: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/decision_repair_path.json`
- Decision-repair path code and figure: `../scripts/analyze_worldagen_decision_repair_path.py`, `../scripts/plot_worldagen_decision_repair_path.py`, `figures/worldagen_decision_repair_path.svg`, `figures/worldagen_decision_repair_path.pdf`
- Candidate-cardinality sensitivity (8 candidates vs fixed linspace-4 subset): `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/candidate_cardinality.json`
- Matched physical-target Goal-invariance audit (162 counterfactuals): `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/goal_invariance_physical_audit.json`
- Physical-target audit code: `../scripts/analyze_worldagen_goal_invariance_physical.py`
- Goal-invariance specification and open validity checks: `../configs/goal_bus_physical_invariance_spec.json`
- Goal-invariance physical-target figure (SVG/PDF): `figures/worldagen_goal_invariance_physical.svg`, `figures/worldagen_goal_invariance_physical.pdf`
- Goal-invariance figure script: `../scripts/plot_worldagen_goal_invariance_physical.py`
- Semantic-surface Goal control: `../runs/worldagen_semantic_goal_invariance_v0/summary.json`
- AlphaBrain direct future-output slot audit (21 fixed-action windows; proprio/wrist/primary target MSE and Goal effect): `../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v1/physical_slot_audit.json`
- AlphaBrain slot audit figure (SVG/PDF): `figures/alphabrain_physical_slot_audit.svg`, `figures/alphabrain_physical_slot_audit.pdf`
- AlphaBrain slot audit scripts: `../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix.py`, `../scripts/aggregate_alphabrain_physical_slot_audit.py`, `../scripts/plot_alphabrain_physical_slot_audit.py`
- Semantic-surface control code: `../scripts/pilot_worldagen_semantic_goal_invariance.py`
- Replay reproducibility boundary audit: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/replay_reproducibility_boundary.json`
- Replay reproducibility audit code: `../scripts/audit_worldagen_replay_reproducibility_boundary.py`
- RNG-persisted reproducible state-target audit (six tasks, three seeds, 144 candidates/288 fresh replay episodes): `../runs/worldagen_reproducible_state_target_all6_replicates_v1/statistics.json`
- Per-seed state-target audits: `../runs/worldagen_reproducible_state_target_all6_seed17/summary.json`, `../runs/worldagen_reproducible_state_target_all6_seed31/summary.json`, `../runs/worldagen_reproducible_state_target_all6_seed37/summary.json`
- RNG-persisted state-target code: `../scripts/pilot_worldagen_reproducible_state_target.py`; aggregate code: `../scripts/aggregate_worldagen_reproducible_state_target.py`
- Candidate-cardinality analysis code: `../scripts/analyze_worldagen_candidate_cardinality.py`
- Forecast-only legal-Goal attacker selection: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/forecast_only_attacker.json`
- Forecast-only attacker analysis code: `../scripts/analyze_forecast_only_goal_attacker.py`
- Latent-divergence legal-Goal attacker selection: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/latent_divergence_attacker.json`
- Latent-divergence attacker analysis code: `../scripts/analyze_latent_divergence_goal_attacker.py`
- Leave-one-seed-out forecast-only attacker result: `../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/oos_attacker.json`
- Leave-one-seed-out attacker analysis code: `../scripts/analyze_worldagen_oos_goal_attacker.py`
- Six-task sweep raw summaries: `../runs/worldagen_aligned_goal_sweep_h30_all6_seed17/summary.json`, `../runs/worldagen_aligned_goal_sweep_h30_all6_seed31/summary.json`, `../runs/worldagen_aligned_goal_sweep_h30_all6_seed37/summary.json`
- Six-task legal Goal dose-response figure (SVG/PDF): `figures/worldagen_aligned_goal_sweep_all6_replicates.svg`, `figures/worldagen_aligned_goal_sweep_all6_replicates.pdf`
- Held-out cross-task value-probe split 1: `../runs/worldagen_cross_task_value_probe_v1/test_tasks679.json`
- Held-out cross-task value-probe split 2: `../runs/worldagen_cross_task_value_probe_v1/test_tasks024.json`
- Cross-task value-probe code: `../scripts/pilot_worldagen_cross_task_value_probe.py`
- WorldAgen forecast-layout contract and regression check: `../scripts/worldagen_forecast_layout.py`, `../scripts/check_worldagen_forecast_layout.py`
- Corrected all-legal-goal sweep statistics: `../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_all_goal_offset_statistics.json`
- Corrected all-legal-goal true-utility audit: `../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_all_goal_evalall_statistics.json`
- Planner-sensitivity six-task matrix: `../runs/worldagen_counterfactual_action_ranking_v0/summary_referencepatch_tasks6.json`
- Aligned scorer-sensitivity comparison: `../runs/worldagen_counterfactual_action_ranking_v0/scorer_sensitivity_tasks6.json`
- Planner-sensitivity all-goal true-utility audit: `../runs/worldagen_counterfactual_action_ranking_v0/referencepatch_all_goal_evalall_statistics.json`
- Bounded learned goal-free context baseline: `../runs/alphabrain_goal_free_context_v1/summary.json`
- Learned context seed-23 replicate: `../runs/alphabrain_goal_free_context_v2/summary.json`
- Learned context multi-timepoint check: `../runs/alphabrain_goal_free_context_v3/summary.json`
- Unified figure: `figures/goal_bus_evidence_summary.svg` or `.pdf`
- Bounded goal-free-context figure: `figures/goal_free_context_baseline.svg` or `.pdf`
- Activation-mediation figure: `figures/worldagen_activation_mediation.svg` or `.pdf`
- Decision evidence figure: `figures/goal_bus_decision_evidence.svg` or `.pdf`
- WorldAgen action-preservation/task-group figure: `figures/worldagen_action_control.svg` or `.pdf`
- Corrected official-preprocess 608-step task pilot: `../runs/alphabrain_cosmos_libero_full_control_v2/spatial_600.json`
- Matching LIBERO-Goal task-0--2 aggregate: `../runs/alphabrain_cosmos_libero_goal_control_v0/aggregate_summary.json`
- AlphaBrain training-resource audit: `../runs/alphabrain_training_resource_audit_v0/summary.json`
- Machine-readable package manifest: `goal_bus_submission_manifest.json`
- Native Action Selection feasibility audit: `../runs/goal_bus_action_selection_audit_v0/summary.json`
- Same-scene Goal crossing (10 legal Goals × 3 shared states × 4 fixed actions): `../runs/worldagen_same_scene_goal_v1/analysis.json`; code: `../scripts/pilot_worldagen_same_scene_goal.py`
- Semantic paraphrase control: `../runs/worldagen_semantic_paraphrase_v0/summary.json`; code: `../scripts/pilot_worldagen_semantic_paraphrase_control.py`
- Same-scene target-error analysis and figure: `../scripts/analyze_worldagen_same_scene_goal.py`, `../scripts/plot_worldagen_same_scene_goal.py`, `figures/worldagen_same_scene_goal_target_delta.svg` / `.pdf`
- Same-scene empty-Goal comparator (50 states × 4 actions, 200 exact replays): `../runs/worldagen_same_scene_empty_goal_all50_v1/summary.json`; figure: `figures/worldagen_empty_goal_comparator.svg`; three-state pilot: `../runs/worldagen_same_scene_empty_goal_v1/summary.json`; code: `../scripts/pilot_worldagen_same_scene_empty_goal.py`
- AlphaBrain six-task independent forecast-driven planner pilot: `../runs/alphabrain_forecast_planner_six_task_v1/summary.json`; code: `../scripts/pilot_alphabrain_forecast_planner.py`

## Reproduction and validation

From the repository root:

```bash
PYTHONPATH=src pytest -q tests
python -m py_compile \
  scripts/analyze_worldagen_offset_stability.py \
  scripts/analyze_alphabrain_forecast_by_task.py \
  scripts/analyze_alphabrain_forecast_matrix.py \
  scripts/plot_goal_bus_evidence_summary.py
PYTHONPATH=src python scripts/summarize.py
```

The existing GPU matrix command is documented in [`goal_bus_reproduction_manifest_cn.md`](goal_bus_reproduction_manifest_cn.md). It uses fixed real LIBERO image/state/action windows and should only be rerun when changing the checkpoint, sampler, or task-selection protocol.

The aligned replicate aggregation is reproducible without GPU after the three run summaries exist:

```bash
.venv-worldagen/bin/python scripts/aggregate_worldagen_aligned_replicates.py \
  runs/worldagen_aligned_planner_v1/summary.json \
  runs/worldagen_aligned_planner_seed31/summary.json \
  runs/worldagen_aligned_planner_seed37/summary.json \
  --output runs/worldagen_aligned_planner_replicates_v1/statistics.json
```

The six-task legal Goal sweep aggregate uses the same scorer and replay protocol:

```bash
.venv-worldagen/bin/python scripts/aggregate_worldagen_aligned_goal_sweep.py \
  runs/worldagen_aligned_goal_sweep_h30_all6_seed17/summary.json \
  runs/worldagen_aligned_goal_sweep_h30_all6_seed31/summary.json \
  runs/worldagen_aligned_goal_sweep_h30_all6_seed37/summary.json \
  --output runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/statistics.json
```

## Interpretation boundary

The evidence supports a cross-carrier prediction-level mechanism and a matched frozen-weight repair. The released WorldAgen and AlphaBrain policies do not use World Forecast for native action selection, so the action-ranking result is explicitly an independent forecast-driven planner experiment. This package does not claim native task-harm rates, closed-loop attack success, or a universally beneficial repair. The two fixed scorers show that Goal-Swap can change real trajectories while the utility direction remains planner-dependent. A bounded learned shared context is included as a forecast comparator, but a task-capable clean checkpoint or a matched trained goal-free dynamics head is still required before extending long-horizon task experiments.

The WorldAgen activation patch adds internal mediation evidence: replacing recorded-goal action-token hidden states in a swapped-goal forward strongly restores the world output at middle layers while leaving the policy branch swapped-goal-sensitive. In the six-task ranking ablation, this forecast restoration recovered the Clean selected action on only 2/6 tasks at either tested layer, so it is a localization diagnostic and a bounded repair probe, not an additional attacker capability or a universal decision fix.

The resource audit records that the released AlphaBrain inference artifacts and official trainer/loader are present, while the local compatible `success_only`/`all_episodes` LIBERO-Cosmos-Policy training dataset is absent. The learned-context result is therefore reported as a bounded comparator rather than a full fine-tuned dynamics baseline.
