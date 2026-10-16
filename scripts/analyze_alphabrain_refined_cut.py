#!/usr/bin/env python3
"""Aggregate the intentionally incomplete world-text-only ablation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_refined_cut_v0'
rows=[json.loads(p.read_text()) for p in sorted(run.glob('window_*.json'))]
world=[float(r['cut_future_world_delta_mean']) for r in rows]
policy=[float(r['cut_policy_delta_mean']) for r in rows]
native=[float(r['native_future_world_delta_mean']) for r in rows]
out={'status':'completed_alphabrain_world_text_only_ablation','cut_mode':'world_text_only','windows':len(rows),'episodes':[r['episode'] for r in rows],'native_future_world_delta_mean':sum(native)/len(native),'world_text_only_cut_future_world_delta_mean':sum(world)/len(world),'world_text_only_cut_future_world_delta_values':world,'world_text_only_cut_future_world_nonzero_all':all(x>0 for x in world),'world_text_only_cut_policy_delta_mean':sum(policy)/len(policy),'finding':'Removing direct text cross-attention from world rows alone does not restore goal invariance: all three windows retain nonzero future-world differences. The residual is consistent with goal carried through contextualized action rows and then read by world rows; the corrected fixed-action world cut removes text from the action/value carriers as well.','scope':'Three real image/state windows, fixed recorded 16-step actions, official two-step sampler; ablation only, not the primary repair.'}
(run/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
