"""Exact forecast-law-preserving rank sampler on a frozen learned world model."""
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
from wm_security.rank_coupling import RankBankSampler
from pilot_mbrl_coupling import make_model,model_hash,save


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    paths=['scripts/pilot_mbrl_rank_coupling.py','src/wm_security/rank_coupling.py','scripts/pilot_mbrl_coupling.py']
    save(out/'plan.json',dict(arguments=vars(args),hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
         stage='Third explicit construction after two Gaussian approximations failed; fresh predetermined seeds 704/705.',
         model='Unchanged v0 native five-member GaussianMLP; no retraining.',
         operation='K fresh complete native forecast batches per candidate; bank-dependent cyclic rank permutations with a fresh uniform row. Marginal preservation is exact by conditional uniform selection.',
         attack_objective='Minimize equal-weight mean-bank quality of the native top-ranked candidate and its top-10 elite set; two coordinate sweeps over K cyclic shifts.',
         matched_control='Independent row per candidate from the identical K-batch generation procedure. Both conditions spend K times the original model simulation compute.',
         cost_boundary='The victim still receives one N-particle estimate per action; the sampler internally generates K such estimates. This extra attacker/control compute is recorded, not called free.',
         scope='Native model, ModelEnv rollouts and CEM with a researcher-defined sampler service intervention; no deployment vulnerability claim.',
         api_requests=0,gpu_work=False))
    model=make_model();model.load(Path(args.model));model.eval()
    before=model_hash(model);rows=[];start=time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed);env=CartPoleEnv();obs,_=env.reset(seed)
        model_env=ModelEnv(env,model,termination_fns.cartpole,reward_fns.cartpole,generator=torch.Generator().manual_seed(seed+10000))
        cfg=OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',num_iterations=3,elite_ratio=.1,
                                 population_size=100,alpha=.1,device='cpu',return_mean_elites=False))
        agent=TrajectoryOptimizerAgent(cfg,[-1.],[1.],planning_horizon=15,replan_freq=1)
        sampler=RankBankSampler(model_env,args.particles,args.bank_size,args.condition,seed+30000)
        agent.set_trajectory_eval_fn(sampler.evaluate)
        episode_start=time.monotonic();trajectory=[];total=0.
        for step in range(200):
            action=agent.act(obs);nxt,reward,done,truncated,_=env.step(action)
            trajectory.append(dict(step=step,obs=obs.tolist(),action=action.tolist(),next_obs=nxt.tolist(),reward=reward,terminated=done))
            total+=reward;obs=nxt
            if (step+1)%25==0:print(json.dumps(dict(condition=args.condition,seed=seed,step=step+1,seconds=time.monotonic()-episode_start)),flush=True)
            if done or truncated:break
        env.close();assert model_hash(model)==before
        row=dict(condition=args.condition,seed=seed,return_=total,steps=len(trajectory),
                 survived_200=len(trajectory)==200 and not trajectory[-1]['terminated'],
                 seconds=time.monotonic()-episode_start,model_hash_unchanged=True,
                 evaluations=sampler.calls,predictive_transitions=sampler.predictive_transitions,
                 phase_candidates=sampler.phase_candidates,
                 surrogate_before_mean=sampler.surrogate_before/max(1,sampler.calls),
                 surrogate_after_mean=sampler.surrogate_after/max(1,sampler.calls))
        rows.append(row);save(out/('episode_'+str(seed)+'.json'),dict(summary=row,trajectory=trajectory));print(json.dumps(row),flush=True)
    result=dict(status='completed',condition=args.condition,rows=rows,model_hash=before,
                mean_return=float(np.mean([r['return_'] for r in rows])),native_environment_steps=sum(r['steps'] for r in rows),
                predictive_transitions=sum(r['predictive_transitions'] for r in rows),
                elapsed_seconds=time.monotonic()-start,api_requests=0,gpu_work=False,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    p.add_argument('--particles',type=int,default=10);p.add_argument('--bank-size',type=int,default=8)
    p.add_argument('--condition',choices=['independent','shared','targeted'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[704,705]);main(p.parse_args())
