"""Native MBRL-Lib gate for temporal-correlation-only imagination changes."""
from __future__ import annotations

import argparse, datetime, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns, termination_fns
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from wm_security.temporal_coupling import TemporalCoupler
from pilot_mbrl_coupling import make_model, model_hash, save

ROOT = Path(__file__).resolve().parents[1]


def evaluate(args, out):
    model = make_model(); model.load(Path(args.model)); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    digest = model_hash(model); rows=[]; started=time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed); env=CartPoleEnv(); obs,_=env.reset(seed)
        model_env=ModelEnv(env, model, termination_fns.cartpole, reward_fns.cartpole,
                           generator=torch.Generator().manual_seed(seed+10000))
        cfg=OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',num_iterations=3,
                                  elite_ratio=.1,population_size=100,alpha=.1,device='cpu',
                                  return_mean_elites=False))
        agent=TrajectoryOptimizerAgent(cfg,[-1.],[1.],planning_horizon=15,replan_freq=1)
        coupler=None
        if args.condition=='persistent':
            coupler=TemporalCoupler(model_env,'persistent'); coupler.install()
        agent.set_trajectory_eval_fn(lambda initial, actions: model_env.evaluate_action_sequences(
            actions, initial, args.particles))
        total=0.;trajectory=[]; t0=time.monotonic()
        try:
            for step in range(200):
                action=agent.act(obs); nxt,reward,done,truncated,_=env.step(action)
                trajectory.append({'step':step,'action':action.tolist(),'reward':float(reward),
                                   'terminated':bool(done),'obs':obs.tolist(),'next_obs':nxt.tolist()})
                total += float(reward); obs=nxt
                if done or truncated: break
        finally:
            if coupler: coupler.restore()
            env.close()
        assert model_hash(model)==digest
        row={'condition':args.condition,'seed':seed,'return_':total,'steps':len(trajectory),
             'survived_200':len(trajectory)==200 and not trajectory[-1]['terminated'],
             'seconds':time.monotonic()-t0,'model_hash_unchanged':True}
        if coupler:
            row.update(marginal_rows=coupler.marginal_rows,
                       temporal_calls=coupler.total_steps,
                       empirical_residual_product_mean=(coupler.autocorrelation_sum /
                                                        max(1,coupler.autocorrelation_count)),
                       used_residual_mean=(coupler.used_sum / max(1,coupler.used_count)),
                       used_residual_std=float(max(0.0, coupler.used_sq_sum / max(1,coupler.used_count) -
                                                   (coupler.used_sum / max(1,coupler.used_count))**2) ** 0.5),
                       used_residual_count=coupler.used_count)
        rows.append(row); save(out/f'episode_{seed}.json',{'summary':row,'trajectory':trajectory})
        print(json.dumps(row),flush=True)
    result={'status':'completed','condition':args.condition,'rows':rows,'model_hash':digest,
            'mean_return':float(np.mean([r['return_'] for r in rows])),
            'native_environment_steps':sum(r['steps'] for r in rows),
            'elapsed_seconds':time.monotonic()-started,'training_updates':0,'api_requests':0,
            'gpu_work':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(out/'summary.json',result);print(json.dumps(result),flush=True)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/pilot_mbrl_temporal_coupling.py','src/wm_security/temporal_coupling.py',
           'scripts/pilot_mbrl_coupling.py']
    save(out/'plan.json',{'arguments':vars(args),'hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
         'model':'Frozen five-member native MBRL-Lib GaussianMLP trained in mbrl_coupling_cartpole_v0.',
         'operation':'Reuse one standard-normal residual at every imagined time step; unconditional residual marginals remain N(0,I), but conditional transition laws need not be preserved.',
         'control':'Upstream MBRL-Lib CEM with the identical model, particles, horizon, population, iterations and environment seeds.',
         'scope':'Two-seed CartPole gate; no claim of novelty, deployment access, or publication-scale evidence.',
         'training_updates':0,'api_requests':0,'gpu_work':False})
    evaluate(args,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    p.add_argument('--condition',choices=['native','persistent'],required=True);p.add_argument('--particles',type=int,default=10)
    p.add_argument('--seeds',nargs='+',type=int,default=[800,801]);main(p.parse_args())
