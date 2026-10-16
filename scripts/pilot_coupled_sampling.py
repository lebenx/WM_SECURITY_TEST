"""CPU security-mechanism gate on untouched Gymnasium FrozenLake dynamics.

The victim is researcher-implemented one-step sample-average planning with exact
optimal continuation values. The predictive kernel is the native kernel itself.
This is NOT a learned-world-model benchmark or an upstream agent reproduction.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from wm_security.coupled_sampling import (particle_law,coupled_policy,independent_policy,
                                        optimize_phases,select,draw_particle_indices)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def arrays(env):
    ns=env.observation_space.n;na=env.action_space.n
    transition=np.zeros((ns,na,ns));reward=np.zeros((ns,na))
    for s in range(ns):
        for a in range(na):
            for p,n,r,done in env.unwrapped.P[s][a]:
                reward[s,a]+=p*r
                if not done:transition[s,a,n]+=p
    return transition,reward


def optimal_values(transition,reward,gamma):
    v=np.zeros(len(transition))
    for _ in range(10000):
        q=reward+gamma*np.einsum('san,n->sa',transition,v);new=q.max(axis=1)
        if np.max(np.abs(new-v))<1e-13:
            return new,reward+gamma*np.einsum('san,n->sa',transition,new)
        v=new
    raise RuntimeError('Value iteration did not converge')


def exact_evaluation(policy,transition,reward,gamma,limit):
    p=np.einsum('sa,san->sn',policy,transition);r=(policy*reward).sum(axis=1)
    v=np.zeros(len(p));g=np.zeros(len(p))
    for _ in range(limit):v=r+gamma*p@v;g=r+p@g
    return dict(success_probability=float(g[0]),discounted_return=float(v[0]),time_limit=limit)


def native_episodes(env,laws,phases,condition,episodes,seed):
    total_steps=0;wins=0;returns=[];imagined_particles=0
    for episode in range(episodes):
        state,_=env.reset(seed=seed+episode);rng=np.random.default_rng(seed+100000+episode)
        done=False;ret=0.;t=0
        while not done:
            uniforms=rng.random(4) if condition=='independent' else np.mod(rng.random()+phases[state],1.)
            # Materialize full particle vectors and compute the actual sample average.
            values=[]
            for law,u in zip(laws[state],uniforms):
                idx=draw_particle_indices(law,u,rng)
                values.append(round(float(law['outcomes'][idx].mean()),12))
                imagined_particles+=len(idx)
            action=int(np.argmax(values))
            state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated;ret+=.99**t*reward;t+=1;wins+=int(reward>0)
        total_steps+=t;returns.append(ret)
    return dict(episodes=episodes,successes=wins,success_rate=wins/episodes,
                discounted_return_mean=float(np.mean(returns)),native_steps=total_steps,
                predictive_particle_draws=imagined_particles)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();gamma=.99
    paths=['scripts/pilot_coupled_sampling.py','src/wm_security/coupled_sampling.py']
    plan=dict(candidate_claim_cn='世界模型的逐动作预测可以保持正确，规划却能被预测样本间的随机耦合操纵。',
        gymnasium=importlib.metadata.version('gymnasium'),maps=args.maps,particles=args.particles,
        gamma=gamma,seed=17,native_episodes=args.episodes,episode_seed=41000,
        model='Exact unmodified native transition kernel; no parameter fitting or prediction corruption.',
        victim='Researcher-defined sample-average one-step lookahead using exact optimal continuation value; all four native actions.',
        adversary='Controls only cross-action coupling in the batch sampler, via fixed state-dependent cyclic uniform shifts. Cannot modify marginal laws, value function, planner, execution, or true environment.',
        optimization='24 random phase candidates plus zero; two coordinate sweeps over a 16-point grid, minimizing the expected selected true Q*.',
        conditions=['independent','shared_quantile','random_phases','optimized_phases','exact_expectation'],
        preservation='For each action even the joint N-particle law remains iid native transitions. Coupling changes only between candidate actions.',
        limits='Analytic-model mechanism gate on two maps of one toy-text task, not a learned neural WM, native agent, deployable attack, or ICLR-scale result.',
        new_api_requests=0,gpu_work=False,
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    save(out/'plan.json',plan);results=[]
    for map_name in args.maps:
        env=gym.make('FrozenLake-v1',map_name=map_name,is_slippery=True)
        transition,reward=arrays(env);v,q=optimal_values(transition,reward,gamma)
        limit=env.spec.max_episode_steps;ns=len(v);optimal=np.eye(4)[q.argmax(axis=1)]
        for particles in args.particles:
            state_laws={};attack={};random={};policies={k:np.zeros_like(q) for k in plan['conditions']}
            policies['exact_expectation']=optimal;counts=0;max_error=0.;moment_error=0.
            rng=np.random.default_rng(17)
            for state in range(ns):
                if env.unwrapped.desc.ravel()[state] in (b'H',b'G'):
                    for k in policies:policies[k][state,0]=1
                    continue
                laws=[]
                for action in range(4):
                    native=env.unwrapped.P[state][action]
                    outcomes=[r+(0 if done else gamma*v[n]) for p,n,r,done in native]
                    law=particle_law(outcomes,[p for p,n,r,done in native],particles);laws.append(law)
                    moment_error=max(moment_error,abs(float(law['weights']@law['values'])-q[state,action]))
                state_laws[state]=laws
                phases,info=optimize_phases(laws,q[state],seed=17+state)
                attack[state]=phases;random[state]=rng.random(4);counts+=info['evaluations']
                policies['independent'][state]=independent_policy(laws)
                for name,z in [('shared_quantile',np.zeros(4)),('random_phases',random[state]),('optimized_phases',phases)]:
                    policies[name][state],err=coupled_policy(laws,z,check_marginals=True)
                    max_error=max(max_error,err)
            assert max_error<1e-10 and moment_error<1e-10
            evaluation={k:exact_evaluation(p,transition,reward,gamma,limit) for k,p in policies.items()}
            native={}
            for name in ['independent','optimized_phases']:
                native[name]=native_episodes(env,state_laws,attack,name,args.episodes,41000)
            folder=out/(map_name+'_n'+str(particles));folder.mkdir()
            save(folder/'phases.json',dict(optimized={s:z.tolist() for s,z in attack.items()},random={s:z.tolist() for s,z in random.items()}))
            save(folder/'policies.json',{k:p.tolist() for k,p in policies.items()})
            row=dict(map=map_name,particles=particles,exact=evaluation,native=native,
                     maximum_count_law_error=max_error,maximum_expected_value_error=moment_error,
                     coupling_objective_evaluations=counts)
            results.append(row);save(folder/'summary.json',row)
            print(json.dumps(row),flush=True)
        env.close()
    summary=dict(status='completed',run=str(out),results=results,
        native_environment_steps=sum(r['native'][k]['native_steps'] for r in results for k in r['native']),
        predictive_particle_draws=sum(r['native'][k]['predictive_particle_draws'] for r in results for k in r['native']),
        new_api_requests=0,neural_training_updates=0,gpu_work=False,
        elapsed_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=plan['limits'])
    save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--maps',nargs='+',default=['4x4','8x8'])
    p.add_argument('--particles',nargs='+',type=int,default=[1,4,16])
    p.add_argument('--episodes',type=int,default=200)
    main(p.parse_args())
