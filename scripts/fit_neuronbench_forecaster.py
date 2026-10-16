"""Fit a shared voltage-based forecaster, then score the frozen model natively."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from neuronbench import worlds, evaluator
from wm_security.neuron_forecaster import fit, trajectory, model


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def parity(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    # Arbitrary non-registry hypotheses; these use no hidden benchmark models.
    parameters=[2.1,95.,-47.,7.,8.,-78.,6.,65.]
    for slow,channel,sign,power,inh in [(False,False,1,1,False),(True,False,1,1,False),
                                      (False,True,1,2,True),(False,True,-1,1,False)]:
        m=model(parameters,channel=channel,slow=slow,sign=sign,power=power,inactivation=inh)
        extras=[]
        if channel:
            extras=[worlds.Chan('parity_hypothesis',g=parameters[0],E=parameters[1],mvh=parameters[2],
                mk=parameters[3]*sign,mtau=parameters[4],mpow=power,hvh=parameters[5] if inh else None,
                hk=parameters[6],htau=parameters[7])]
        for label,segments in [('single',[(80,10.)]),('sequence',[(100,-20.),(80,12.)])]:
            I,ts=worlds.build_I(segments)
            n,V=worlds.simulate(I,ts,extra=extras,slow_na=slow,trace=True)
            fast_n,fast_V,_=trajectory(m,segments)
            row=dict(slow=slow,channel=channel,sign=sign,power=power,inactivation=inh,
                     protocol=label,native_count=n,accelerated_count=fast_n,
                     max_voltage_error=float(np.max(np.abs(V-fast_V))))
            rows.append(row)
            save(out/'parity.json',dict(rows=rows,complete=False))
    result=dict(complete=True,comparisons=len(rows),rows=rows,
                native_hypothesis_simulations=len(rows),accelerated_hypothesis_simulations=len(rows),
                true_cell_queries=0,passed=all(r['native_count']==r['accelerated_count'] and r['max_voltage_error']<1e-6 for r in rows))
    save(out/'parity.json',result);print(json.dumps(result),flush=True)
    assert result['passed']


def main(args):
    if args.parity:
        parity(args.output);return
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    records=[json.loads(s) for s in (source/'observations.jsonl').read_text().splitlines()]
    # Strip labels and all task identity before handing observations to the learner.
    observed=[{k:r[k] for k in ['segments','voltage','obs_idx']} for r in records]
    module=Path('src/wm_security/neuron_forecaster.py')
    save(out/'plan.json',dict(source=str(source),source_sha256=hashlib.sha256((source/'observations.jsonl').read_bytes()).hexdigest(),
        learner_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),seed=17,starts_per_structure=16,
        refined_starts_per_structure=2,max_nfev=120,
        model_class='Public plain/slow-Na families; one generic extra current with continuous parameters, activation sign +/- and power1/2, optional inactivation.',
        training='Fit voltage-synchronized residuals, select using unsynchronized training voltage RMSE.',
        held_out_outcomes_used_for_fit=False,new_oracle_queries=0,new_api_requests=0,gpu_work=False,
        scope='Fixed-family system-identification comparator; not new method, open-ended mechanism discovery, or full MDA reproduction.'))
    start=time.monotonic()
    def callback(candidate,counts):
        with (out/'fit_candidates.jsonl').open('a') as log:log.write(json.dumps(candidate)+'\n')
        print(json.dumps(dict(kind=candidate['kind'],sign=candidate['model']['sign'],power=candidate['model']['power'],
            inactivation=candidate['model']['inactivation'],training_rmse=candidate['training_rmse'],
            synchronized_rmse=candidate['synchronized_rmse'],nfev=candidate['nfev'],simulations=counts)),flush=True)
    result=fit(observed,callback=callback)
    save(out/'fitted_model.json',result)
    fit_seconds=time.monotonic()-start
    # Forecast metadata and labels only enter after the model has been frozen.
    old=json.loads((source/'summary.json').read_text())
    import neuronbench as nb
    world=nb.load_world(old['world'],stochastic=False,seed=0)
    counts={};traces={}
    for label,segments in world.test_protocols:
        cnt,V,ts=trajectory(result['selected']['model'],segments)
        counts[label]=cnt;traces[label]=V
    # Reuse previously scored count targets, avoiding fresh truth queries for this metric.
    mse=evaluator.forecast_mse(counts,old['held_out_targets'])
    feature_mse=world.feature_forecast_mse(traces)
    np.savez_compressed(out/'predicted_traces.npz',**{f'trace_{i}':v for i,v in enumerate(traces.values())})
    summary=dict(status='completed',source=str(source),run=str(out),world=old['world'],
        training_rmse=result['selected']['training_rmse'],selected_model=result['selected']['model'],
        predictions=counts,native_spike_forecast_mse=mse,native_feature_forecast_mse=feature_mse,
        previous_llm_spike_mse=old['native_llm_spike_forecast_mse'],
        previous_reference_spike_mse=old['plain_reference_spike_forecast_mse'],
        model_simulations=result['counts'],forecast_model_simulations=len(counts),
        held_out_feature_scoring_trajectories=len(counts),new_oracle_experiments=0,source_experiments_reused=len(observed),
        new_api_requests=0,gpu_work=False,fit_seconds=fit_seconds,total_seconds=time.monotonic()-start,
        scope='Fixed-family mechanistic fit on existing noiseless voltage; no new experiment constructor or full MDA reproduction.',
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--output',required=True)
    p.add_argument('--parity',action='store_true');main(p.parse_args())
