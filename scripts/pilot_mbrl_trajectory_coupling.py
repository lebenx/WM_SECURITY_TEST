"""A frozen-model follow-up: whole-rollout coupling on fresh CartPole seeds."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns,termination_fns
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from wm_security.trajectory_coupling import TrajectoryCoupler
from pilot_mbrl_coupling import make_model,model_hash,save


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    paths=['scripts/pilot_mbrl_trajectory_coupling.py','src/wm_security/trajectory_coupling.py','scripts/pilot_mbrl_coupling.py']
    save(out/'plan.json',dict(arguments=vars(args),hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
         stage='Whole-trajectory follow-up after stepwise targeted heuristic failed; fresh predetermined seeds 702/703.',
         model='Same frozen five-member learned GaussianMLP checkpoint as v0; no new training or validation-based model selection.',
         operation='Project full H*N*D noise onto the model-derived smooth-survival sensitivity, then replace its standard-normal coordinate with a coupled normal quantile. Choose phases from 20 fixed candidates using a 64-point surrogate integration grid.',
         scope='Native MBRL-Lib dynamics, ModelEnv rollout and CEM; attacker surrogate affects only noise coupling. Not a deployment vulnerability or full PETS reproduction.',
         api_requests=0,gpu_work=False))
    model=make_model();model.load(Path(args.model));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    before=model_hash(model);rows=[];start=time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed);env=CartPoleEnv();obs,_=env.reset(seed)
        model_env=ModelEnv(env,model,termination_fns.cartpole,reward_fns.cartpole,generator=torch.Generator().manual_seed(seed+10000))
        cfg=OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',num_iterations=3,elite_ratio=.1,
                                 population_size=100,alpha=.1,device='cpu',return_mean_elites=False))
        agent=TrajectoryOptimizerAgent(cfg,[-1.],[1.],planning_horizon=15,replan_freq=1)
        coupling=None
        if args.condition!='native':
            coupling=TrajectoryCoupler(model,model_env,args.particles,args.condition,seed+20000);coupling.install()
        calls=0
        def evaluator(initial_obs,actions):
            nonlocal calls
            calls+=1
            if coupling:coupling.actions=actions
            return model_env.evaluate_action_sequences(actions,initial_obs,args.particles)
        agent.set_trajectory_eval_fn(evaluator)
        episode_start=time.monotonic();trajectory=[];total=0.
        try:
            for step in range(200):
                action=agent.act(obs);nxt,reward,done,truncated,_=env.step(action)
                trajectory.append(dict(step=step,obs=obs.tolist(),action=action.tolist(),next_obs=nxt.tolist(),reward=reward,terminated=done))
                total+=reward;obs=nxt
                if (step+1)%25==0:print(json.dumps(dict(condition=args.condition,seed=seed,step=step+1,seconds=time.monotonic()-episode_start)),flush=True)
                if done or truncated:break
        finally:
            if coupling:coupling.restore()
            env.close()
        assert model_hash(model)==before
        row=dict(condition=args.condition,seed=seed,return_=total,steps=len(trajectory),
                 survived_200=len(trajectory)==200 and not trajectory[-1]['terminated'],
                 seconds=time.monotonic()-episode_start,model_hash_unchanged=True,
                 evaluations=calls,predictive_transitions=calls*100*15*args.particles,
                 surrogate_forward_transitions=coupling.surrogate_forward_transitions if coupling else 0,
                 copula_candidates_evaluated=coupling.copula_candidates if coupling else 0)
        rows.append(row);save(out/('episode_'+str(seed)+'.json'),dict(summary=row,trajectory=trajectory));print(json.dumps(row),flush=True)
    result=dict(status='completed',condition=args.condition,rows=rows,model_hash=before,
                mean_return=float(np.mean([r['return_'] for r in rows])),native_environment_steps=sum(r['steps'] for r in rows),
                predictive_transitions=sum(r['predictive_transitions'] for r in rows),
                surrogate_forward_transitions=sum(r['surrogate_forward_transitions'] for r in rows),
                elapsed_seconds=time.monotonic()-start,api_requests=0,gpu_work=False,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    p.add_argument('--particles',type=int,default=10)
    p.add_argument('--condition',choices=['native','shared','spread','independent','targeted'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[702,703]);main(p.parse_args())
