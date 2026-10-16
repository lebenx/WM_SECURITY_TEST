#!/usr/bin/env python3
"""Plot the existing goal-bus offset and task-stratified aggregates."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
offset=json.loads((ROOT/'runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json').read_text())
task=json.loads((ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_by_task_v0/aggregate.json').read_text())
outdir=ROOT/'reports/figures'; outdir.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),gridspec_kw={'width_ratios':[1.05,1.35]})
# Offset panel
ax=axes[0]; offs=np.array(offset['offsets']); native=np.array([offset['by_offset'][str(o)]['native_world_relative_l2_mean'] for o in offs]); cut=np.zeros_like(native); pol=np.array([offset['by_offset'][str(o)]['cut_policy_relative_l2_mean'] for o in offs])
ax.plot(offs,native,'o-',color='#b23a48',lw=2,ms=5,label='native world')
ax.plot(offs,cut,'s--',color='#2166ac',lw=1.8,ms=4,label='fixed-action cut world')
ax.plot(offs,pol,'^:',color='#4d9221',lw=1.5,ms=4,label='cut policy')
ax.set_xlabel('Legal goal offset'); ax.set_ylabel('Mean relative L2'); ax.set_title('WorldAgen: 30 windows × 9 offsets')
ax.set_xticks(offs); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,fontsize=8,loc='upper right')
# Task panel
ax=axes[1]; groups=task['groups']; labels=[]; vals=[]
for i,g in enumerate(groups):
 t=g['task_a']; short=(t.replace('put ','').replace(' and close it','').replace(' and put ',' + '))
 if len(short)>34: short=short[:31]+'…'
 labels.append(short); vals.append(g['native_world_delta_mean_rows'])
y=np.arange(len(groups)); ax.barh(y,vals,color='#b23a48',alpha=.9,label='native world'); ax.barh(y,np.zeros(len(vals)),color='#2166ac',alpha=.9,label='fixed-action cut (exact zero)')
ax.set_yticks(y,labels,fontsize=8); ax.invert_yaxis(); ax.set_xlabel('Mean relative L2'); ax.set_title('AlphaBrain: 5 task-A groups\n(seed 1 + seed 3 windows)'); ax.grid(axis='x',alpha=.25); ax.legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('Shared action-token bus: goal sensitivity and matched repair',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(outdir/'goal_bus_evidence_summary.svg',bbox_inches='tight')
fig.savefig(outdir/'goal_bus_evidence_summary.pdf',bbox_inches='tight')
print(outdir/'goal_bus_evidence_summary.svg'); print(outdir/'goal_bus_evidence_summary.pdf')
