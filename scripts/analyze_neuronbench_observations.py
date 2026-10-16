"""Compare already-observed voltage with the public reference; no new true-cell queries."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import neuronbench as nb


def main(args):
    root=Path(args.run);output=[];start=time.monotonic();simulations=0
    for folder in sorted(root.iterdir()):
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        world=nb.load_world(summary['world'],stochastic=False,seed=0)
        for line in (folder/'observations.jsonl').read_text().splitlines():
            obs=json.loads(line);I,ts=nb.worlds.build_I(obs['segments'])
            count,V=nb.worlds.simulate(I,ts,**world.reference_model,trace=True);simulations+=1
            idx=np.array(obs['obs_idx']);actual=np.array(obs['voltage']);reference=V[idx]
            test=obs['test_start'];difference=actual-reference
            output.append(dict(world=summary['world'],query=obs['query'],label=obs['label'],
                observed_spikes=obs['spike_count'],reference_spikes=count,
                spike_count_difference=obs['spike_count']-count,
                voltage_rmse_mV=float(np.sqrt(np.mean(difference**2))),
                voltage_max_abs_difference_mV=float(np.max(np.abs(difference))),
                pretest_voltage_rmse_mV=float(np.sqrt(np.mean(difference[:test]**2))) if test else None,
                note='Noiseless paired reference comparison, not an identifiability proof or a result for the noisy benchmark.'))
    result=dict(status='completed',observations=output,new_oracle_experiments=0,
                public_reference_simulations=simulations,api_requests=0,gpu_work=False,
                elapsed_s=time.monotonic()-start)
    (root/'observation_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args())
