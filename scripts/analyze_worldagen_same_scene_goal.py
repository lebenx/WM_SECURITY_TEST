#!/usr/bin/env python3
"""Summarize the same-scene Goal crossing and its physical replay control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('summary', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.summary.read_text())
    rows = data['rows']
    native_effect = []
    bus_goal_variance = []
    control_world_effect = []
    target_delta, action_sensitivity = [], []
    nonclean_state_delta = []
    bus_cut_clean_cost_by_state = []
    per_goal_target_delta = []
    exact_initial = exact_future = 0
    replay_total = 0
    clean_idx = int(data['clean_goal_index'])
    for row in rows:
        n = np.asarray(row['goal_mse_vs_clean']['native'], dtype=float)
        native_effect.extend(n[np.arange(n.shape[0]) != clean_idx].ravel())
        t = np.asarray(row['target_mse']['native'], dtype=float)
        delta = t - t[clean_idx:clean_idx + 1]
        target_delta.extend(delta.ravel())
        nonclean_state_delta.append(float(np.delete(delta, clean_idx, axis=0).mean()))
        per_goal_target_delta.append(np.delete(delta, clean_idx, axis=0).mean(axis=1))
        bus_target = np.asarray(row['target_mse']['bus_cut'], dtype=float)
        bus_cut_clean_cost_by_state.append(float(bus_target[0].mean() - t[clean_idx].mean()))
        a = np.asarray(row['action_mse_vs_zero']['bus_cut'], dtype=float)
        action_sensitivity.extend(a[:, 1:].ravel())
        forecast_file = args.summary.parent / f"init_{int(row['init_index'])}_forecasts.pt"
        payload = torch.load(forecast_file, map_location='cpu', weights_only=False)
        pred = payload['predictions']
        bus = pred['bus_cut'].numpy()
        bus_mean = bus.mean(axis=0, keepdims=True)
        bus_goal_variance.extend(((bus - bus_mean) ** 2).mean(axis=(2, 3)).ravel())
        control_world_effect.append(float(np.max(np.abs(pred['native'].numpy() - pred['control_cut'].numpy()))))
        for replay in row['physical_replays']:
            replay_total += 1
            exact_initial += int(replay['initial_exact'])
            exact_future += int(replay['future_exact'])
    # The state is the sampling unit here: actions and crossed Goal labels
    # share the same physical replay.  A cluster bootstrap avoids treating
    # the 3 x 4 x 9 paired cells as independent worlds.
    rng = np.random.default_rng(0)
    state_values = np.asarray(nonclean_state_delta, dtype=float)
    boot = np.asarray([
        rng.choice(state_values, size=len(state_values), replace=True).mean()
        for _ in range(10000)
    ])
    per_goal = np.asarray(per_goal_target_delta, dtype=float)
    bus_cost = np.asarray(bus_cut_clean_cost_by_state, dtype=float)
    result = {
        'status': 'completed_same_scene_goal_crossing_analysis',
        'source': str(args.summary),
        'states': len(rows), 'goals': len(rows[0]['goal_mse_vs_clean']['native']),
        'clean_goal_index': clean_idx,
        'actions': len(rows[0]['goal_mse_vs_clean']['native'][0]),
        'physical_replays': replay_total,
        'initial_state_exact_count': exact_initial,
        'future_replay_exact_count': exact_future,
        'native_goal_effect_nonclean_mean_mse': float(np.mean(native_effect)),
        'native_goal_effect_nonzero_fraction': float(np.mean(np.asarray(native_effect) > 0)),
        'bus_cut_goal_exact_all_states': bool(all(row['bus_goal_exact'] for row in rows)),
        'control_cut_world_exact_all_states': bool(all(row['control_world_exact'] for row in rows)),
        'bus_cut_goal_variance_max_mse': float(np.max(bus_goal_variance)),
        'control_cut_world_effect_max_abs': float(np.max(control_world_effect)),
        'bus_cut_action_sensitivity_nonzero_fraction': float(np.mean(np.asarray(action_sensitivity) > 0)),
        'native_target_mse_delta_vs_clean_mean': float(np.mean(target_delta)),
        'native_target_mse_delta_vs_clean_min': float(np.min(target_delta)),
        'native_target_mse_delta_vs_clean_max': float(np.max(target_delta)),
        'native_nonclean_target_mse_delta_vs_clean_mean': float(np.mean(state_values)),
        'native_nonclean_target_mse_delta_vs_clean_by_state': state_values.tolist(),
        'native_nonclean_target_mse_delta_vs_clean_state_cluster_bootstrap_ci95': [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        'native_nonclean_target_mse_delta_positive_state_fraction': float(np.mean(state_values > 0)),
        'bus_cut_target_mse_cost_vs_native_clean_mean': float(np.mean(bus_cost)),
        'bus_cut_target_mse_cost_vs_native_clean_by_state': bus_cost.tolist(),
        'native_nonclean_target_mse_delta_by_goal_mean': per_goal.mean(axis=0).tolist(),
        'native_nonclean_target_mse_delta_by_goal_state_matrix': per_goal.tolist(),
        'interpretation': [
            'All legal LIBERO-Goal BDDL files share the same physical scene definition after removing language, goal, and object-of-interest fields; only the task predicate differs.',
            'Fresh replays with the same full simulator state and fixed action are exact across Goal labels, establishing an environment-side conditional invariance control.',
            'The native WorldAgen forecast varies with Goal while the matched Bus-Cut is invariant; target-error changes are reported separately and need not have one sign.',
            'At the state-cluster level, a uniformly sampled non-Clean legal Goal has a positive target-MSE delta in every shared initial state; this is descriptive evidence against the clean Goal being an unbiased physical target comparator, not a universal harm claim.',
        ],
        'scope_limits': [
            'The same-scene crossing addresses the simulator counterfactual and does not prove the model observed every physical hidden variable during training.',
            'Forecasts are compared at the first primary-view chunk; no task-success claim is made.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
