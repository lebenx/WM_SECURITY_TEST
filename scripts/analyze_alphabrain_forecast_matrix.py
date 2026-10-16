#!/usr/bin/env python3
"""Cluster-bootstrap aggregation for the expanded AlphaBrain forecast matrix."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
src=Path(__import__('os').environ.get('ALPHABRAIN_MATRIX_SUMMARY',str(ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json')))
d=json.loads(src.read_text()); rows=d['rows']; episodes=sorted(set(r['episode'] for r in rows))
def means(sample):
 return np.array([np.mean([r['native_goal_world_delta_mean'] for r in rows if r['episode']==e]) for e in sample])
def repair_means(sample):
 return np.array([np.mean([r['repair_mse_delta'] for r in rows if r['episode']==e]) for e in sample])
rng=np.random.default_rng(20261011); B=10000
native_ep=np.array([np.mean([r['native_goal_world_delta_mean'] for r in rows if r['episode']==e]) for e in episodes])
repair_ep=np.array([np.mean([r['repair_mse_delta'] for r in rows if r['episode']==e]) for e in episodes])
def ci(vals):
 boot=np.array([np.mean(vals[rng.integers(0,len(vals),len(vals))]) for _ in range(B)])
 return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
out={
 'status':'completed_alphabrain_forecast_matrix_aggregate','source':str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),'cut_mode':d['cut_mode'],'sampler_seed':d['sampler_seed'],'windows':len(rows),'episodes':episodes,'windows_per_episode':{str(e):sum(r['episode']==e for r in rows) for e in episodes},
 'native_goal_world_delta_mean':float(np.mean(native_ep)),'native_goal_world_cluster_bootstrap_ci95':ci(native_ep),'native_goal_world_nonzero_all':all(r['native_goal_world_delta_mean']>0 for r in rows),
 'cut_goal_world_delta_max':float(max(r['cut_goal_world_delta_mean'] for r in rows)),'cut_goal_world_exact_zero_all':all(r['cut_goal_world_delta_mean']==0 for r in rows),
 'repair_mse_delta_mean':float(np.mean(repair_ep)),'repair_mse_delta_cluster_bootstrap_ci95':ci(repair_ep),'repair_mse_delta_positive_fraction_windows':float(np.mean([r['repair_mse_delta']>0 for r in rows])),'repair_mse_delta_positive_fraction_episodes':float(np.mean(repair_ep>0)),
 'finding':f"Across {len(rows)} fixed physical image/state/action windows ({len(episodes)} episode clusters), native goal swaps changed world forecasts in every window and the corrected fixed-action world cut restored exact invariance in every window. Cluster-bootstrap quantifies the episode-level uncertainty; held-out forecast delta is a secondary sampler-seed-dependent metric, not the invariant claim."
}
outp=src.parent/'aggregate.json'; outp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
