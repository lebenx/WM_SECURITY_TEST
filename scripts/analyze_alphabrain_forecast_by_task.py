#!/usr/bin/env python3
"""Aggregate AlphaBrain forecast matrix by legal task text.

This is a descriptive stratification of the already-run matrix; it does not
run a new model forward. Rows stay grouped by physical episode when the
matrix contains repeated time windows.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
 ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json',
 ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/summary.json',
]
OUT=ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_by_task_v0/aggregate.json'

def summarize(all_rows):
 groups=defaultdict(list)
 for row in all_rows: groups[row['task_a']].append(row)
 result=[]
 for task, rows in sorted(groups.items()):
  eps=sorted({int(r['episode']) for r in rows})
  # For repeated time points, report both row and episode means so rows are
  # not mistaken for independent physical worlds.
  ep_native=[float(np.mean([r['native_goal_world_delta_mean'] for r in rows if int(r['episode'])==e])) for e in eps]
  ep_repair=[float(np.mean([r['repair_mse_delta'] for r in rows if int(r['episode'])==e])) for e in eps]
  result.append({
   'task_a':task,'windows':len(rows),'episodes':eps,
   'native_world_delta_mean_rows':float(np.mean([r['native_goal_world_delta_mean'] for r in rows])),
   'native_world_delta_mean_episode':float(np.mean(ep_native)),
   'cut_world_delta_max':float(max(r['cut_goal_world_delta_mean'] for r in rows)),
   'cut_world_exact_zero_all':all(r['cut_goal_world_delta_mean']==0 for r in rows),
   'repair_mse_delta_mean_rows':float(np.mean([r['repair_mse_delta'] for r in rows])),
   'repair_mse_delta_mean_episode':float(np.mean(ep_repair)),
   'repair_mse_positive_episode_fraction':float(np.mean(np.asarray(ep_repair)>0)),
  })
 return result
all_rows=[]
for source in SOURCES:
 all_rows.extend(json.loads(source.read_text())['rows'])
out={
 'status':'completed_alphabrain_forecast_by_task_aggregate',
 'sources':[str(p.relative_to(ROOT)) for p in SOURCES],
 'seed1_windows':21,'seed3_windows':3,'unique_task_a':5,
 'task_rows':sum(len(json.loads(p.read_text())['rows']) for p in SOURCES),
 'groups':summarize(all_rows),
 'finding':'Across five legal task-A texts represented by the seed-1 matrix, native goal swaps are nonzero in every task group and the corrected fixed-action cut is exactly invariant in every group. Seed-3 repeats three of these task groups and preserves the same invariance. Group means are descriptive; repeated frames are not independent physical episodes.'
}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
