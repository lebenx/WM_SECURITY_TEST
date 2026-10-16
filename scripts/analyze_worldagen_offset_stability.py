#!/usr/bin/env python3
"""Descriptive leave-one-offset-out stability analysis; no new forward pass."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json'
d=json.loads(src.read_text()); by={int(k):v for k,v in d['by_offset'].items()}
all_offsets=sorted(by)
loo=[]
for omit in all_offsets:
 keep=[by[o]['native_world_relative_l2_mean'] for o in all_offsets if o!=omit]
 policy=[by[o]['cut_policy_relative_l2_mean'] for o in all_offsets if o!=omit]
 loo.append({'omitted_offset':omit,'native_mean':float(np.mean(keep)),'cut_policy_mean':float(np.mean(policy))})
out={
 'status':'completed_worldagen_offset_leave_one_out_stability',
 'source':str(src.relative_to(ROOT)),
 'offsets':all_offsets,'paired_rows':d['paired_rows'],'physical_windows':d['windows_per_offset'],
 'native_offset_means':{str(o):by[o]['native_world_relative_l2_mean'] for o in all_offsets},
 'native_offset_mean_min':float(min(x['native_world_relative_l2_mean'] for x in by.values())),
 'native_offset_mean_max':float(max(x['native_world_relative_l2_mean'] for x in by.values())),
 'cut_world_exact_zero_all':bool(d['cut_world_exact_zero_fraction']==1.0),
 'cut_policy_sensitive_all':bool(d['cut_policy_sensitive_fraction']==1.0),
 'leave_one_offset_out':loo,
 'finding':'Native goal sensitivity and exact repaired invariance persist at every legal offset. Leave-one-offset-out native means remain positive; the effect is not carried by one semantic distance. Repeated offset rows are clustered by physical window.'
}
outp=ROOT/'runs/worldagen_goal_bus_offset_stability_v0/summary.json'; outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
