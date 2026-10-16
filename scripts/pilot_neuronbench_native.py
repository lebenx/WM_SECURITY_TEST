"""Bounded native NeuronBench capability baseline; no new-method claim."""
import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np
import neuronbench as nb
from wm_security.client import Client


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class Adapter:
    def __init__(self, model, out):
        self.model, self.out, self.requests, self.failures = model, out, 0, 0

    def ask(self, system, prompt, temperature=.2, max_tokens=800, response_format=None):
        # Reload user-managed external credentials for every request.
        c=Client(self.model,log_dir=self.out,timeout=60,max_calls=1)
        c.config=dict(c.config,base_url=c.config['fallback_url'],api_key=c.config['fallback_api_key'])
        self.requests+=1
        try:
            return c.chat([{'role':'system','content':system},{'role':'user','content':prompt}],
                          temperature=temperature,max_tokens=max_tokens,
                          json_mode=response_format is not None,tag=f'neuronbench_native_request_{self.requests}')
        except Exception:
            self.failures+=1
            raise


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    world=nb.load_world(args.world,stochastic=False,seed=0)
    spec=world.problem()
    # This is the native baseline's exact menu/count interface, not a proposed mechanism.
    save(out/'public_problem.json',spec)
    save(out/'plan.json',dict(world=args.world,model=args.model,budget=args.budget,
         stochastic=False,seed=0,protocol='Native choose_experiment and forecast_counts; identical menu and prompt behavior.',
         source_pin='c354622458c460b419cab821d482c879f0578377',
         truth_fields_passed_to_model=False,primary_called=False,gpu_work=False,
         scope='Two-world development capability gate; no MDA reproduction, no program generation or full benchmark result.'))
    adapter=Adapter(args.model,out)
    start=time.monotonic();remaining=list(world.protocol_pool);observed=[];records=[]
    for i in range(args.budget):
        protocol=nb.agent.choose_experiment(world,observed,remaining,adapter)
        o=world.run(protocol)
        record=dict(query=i+1,label=protocol[0],segments=protocol[1],spike_count=o.spike_count,
                    cost=o.cost,voltage=o.voltage.tolist(),obs_idx=o.obs_idx.tolist(),test_start=o.test_start)
        records.append(record)
        with (out/'observations.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        observed.append((protocol[0],round(o.spike_count)))
        remaining=[p for p in remaining if p[0]!=protocol[0]]
        print(json.dumps({k:record[k] for k in ['query','label','spike_count','cost']}),flush=True)
    predictions=nb.agent.forecast_counts(world,observed,adapter)
    # Plain model is explicitly part of the public problem and requires no oracle observations.
    reference={lab:float(world.simulate(world.reference_model,(lab,seg)))
               for lab,seg in world.test_protocols}
    # Score once against native targets, then reuse those targets across both forecasts.
    targets=nb.evaluator.held_out_targets(args.world,stochastic=False,seed=0)
    result=dict(status='completed' if not adapter.failures else 'completed_with_native_fallback_after_api_failure',
         world=args.world,model=args.model,run=str(out),observed=observed,predictions=predictions,
         reference_predictions=reference,held_out_targets=targets,
         native_llm_spike_forecast_mse=nb.evaluator.forecast_mse(predictions,targets),
         plain_reference_spike_forecast_mse=nb.evaluator.forecast_mse(reference,targets),
         oracle_experiments=sum(r['cost'] for r in records),api_requests=adapter.requests,
         failed_api_requests=adapter.failures,reference_model_simulations=len(reference),
         held_out_scoring_trajectories=len(targets),
         feature_forecast_mse=None,metric_scope='Native headline spike-count MSE only; the native count-only LLM baseline does not supply voltage trajectories.',
         new_method=False,full_mda_reproduction=False,gpu_work=False,
         elapsed_s=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',required=True)
    p.add_argument('--model',default='glm-5.3');p.add_argument('--budget',type=int,default=3)
    p.add_argument('--output',required=True);main(p.parse_args())
