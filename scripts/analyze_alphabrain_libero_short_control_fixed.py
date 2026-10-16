#!/usr/bin/env python3
"""Aggregate the corrected AlphaBrain fixed-action LIBERO control pilot."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/alphabrain_cosmos_libero_short_control_fixed_action_v1'
files=[run/'spatial.json',run/'object.json',run/'goal.json']
rows=[json.loads(p.read_text()) for p in files]
results=[dict(x, condition=('fixed_action_world_cut' if x['condition']=='edge_cut' else x['condition'])) for d in rows for x in d['results']]
def arm(name):
 xs=[x for x in results if x['condition']==name]
 return {'successes':sum(bool(x['success']) for x in xs),'episodes':len(xs),'success_rate':sum(bool(x['success']) for x in xs)/len(xs),'environment_steps':[x['environment_steps'] for x in xs],'action_norm_mean_mean':sum(x['action_norm_mean'] for x in xs)/len(xs)}
out={'status':'completed_alphabrain_libero_short_control_fixed_action_aggregate','suites':[d['suite'] for d in rows],'conditions':['native','fixed_action_world_cut'],'cut_mode':'fixed_action_world','episodes':len(results),'steps_per_episode':32,'results':results,'aggregate':{'native':arm('native'),'fixed_action_world_cut':arm('fixed_action_world_cut')},'finding':'Across one initial state in each of LIBERO spatial, object, and goal task families, native and corrected fixed-action world cut both had 0/3 successes at the deliberately short 32-step horizon. This is a corrected control-preservation sanity check, not a task-harm or success-rate claim.','scope':'Three task families, one task and one initial state per family, two sampler action chunks per condition; corrected world-branch cut; clean task gate and full-horizon evaluation remain separate.'}
(run/'aggregate_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
