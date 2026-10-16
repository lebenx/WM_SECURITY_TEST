"""Bounded white-box search over genuine action-generated learning curricula.

Candidate trajectories come from a private reference simulator, not edited
transition targets. A chosen whole trajectory can be replayed by actions in the
shared learner's source task. Search costs and deployed actions are separate.
This screen tests feasibility, not novelty or collective persistence.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from wm_security.tdmpc_native import load_agent, episode, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt


def snapshot(agent):
    return {k:v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
            for k,v in agent.model.state_dict().items()}


def reset_agent(agent, original):
    agent.model.load_state_dict(original)
    agent.optim.state.clear();agent.pi_optim.state.clear()
    agent.model.zero_grad(set_to_none=True)
    agent.scale.value.fill_(1.)
    agent._prev_mean.zero_()
    agent.model.eval()


def wave_trajectory(agent, task, seed, theta, steps):
    """Only bounded actuator commands; genuine observations and rewards."""
    cfg = agent.cfg;idx = cfg.tasks.index(task);dim = cfg.action_dims[idx]
    theta = np.asarray(theta).reshape(4, dim)
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    bias = np.tanh(theta[0]);amplitude = 2 * sigmoid(theta[1])
    period = np.exp(np.log(4) + sigmoid(theta[2]) * np.log(20))
    phase = np.pi * np.tanh(theta[3])
    env = make_one(task, seed)
    obs = pad_obs(env.reset(), cfg)
    observations, actions, rewards = [obs], [], []
    try:
        for t in range(steps):
            raw = np.clip(bias + amplitude * np.sin(2*np.pi*t/period + phase), -1, 1)
            action = torch.zeros(cfg.action_dim)
            action[:dim] = torch.tensor(raw, dtype=torch.float32)
            obs, reward, done, info = env.step(action[:dim])
            observations.append(pad_obs(obs, cfg));actions.append(action);rewards.append(float(reward))
            if done:
                break
    finally:
        env.unwrapped.close()
    assert len(actions) == steps
    return {'obs':torch.stack(observations), 'action':torch.stack(actions),
            'reward':torch.tensor(rewards), 'task':idx, 'seed':seed,
            'theta':torch.tensor(theta), 'generation':'bounded sinusoidal actuator commands'}


@torch.no_grad()
def source_error(agent, data):
    """Mean original three-step latent consistency error on all source windows."""
    cfg=agent.cfg;m=agent.model;task=data['task']
    obs=data['obs'].to(agent.device);actions=data['action'].to(agent.device)
    encoded=m.encode(obs, task)
    starts=torch.arange(len(actions)-cfg.horizon+1, device=agent.device)
    z=encoded[starts];loss=0.
    for t in range(cfg.horizon):
        z=m.next(z, actions[starts+t], task)
        loss=loss+cfg.rho**t*F.mse_loss(z, encoded[starts+t+1])
    return float(loss/cfg.horizon)


@torch.no_grad()
def make_probe(agent, reference):
    """16 calibration states x32 fixed three-step candidate sequences."""
    set_seed(731)
    cfg=agent.cfg
    indexes=torch.linspace(50,len(reference['action'])-cfg.horizon-1,16).long()
    obs=reference['obs'][indexes].repeat_interleave(32,dim=0).to(agent.device)
    behavior=torch.stack([reference['action'][indexes+t] for t in range(cfg.horizon)])
    actions=behavior[:, :, None, :].repeat(1,1,32,1).to(agent.device)
    actions[:,:,1:16]+=torch.randn_like(actions[:,:,1:16])*0.3
    actions[:,:,16:]=torch.rand_like(actions[:,:,16:])*2-1
    actions=actions.clamp(-1,1).reshape(cfg.horizon,512,cfg.action_dim)
    actions*=agent.model._action_masks[reference['task']]
    return {'obs':obs,'actions':actions,'task':reference['task'],'indexes':indexes.tolist()}


@torch.no_grad()
def probe_values(agent, probe):
    # Original trajectory-value estimator, same bootstrap/Q sampling RNG.
    set_seed(991)
    task=torch.tensor([probe['task']],device=agent.device)
    z=agent.model.encode(probe['obs'],task)
    values=agent._estimate_value(z,probe['actions'],task)
    return values.reshape(16,32).detach()


def score_values(values, clean):
    choice=values.argmax(1)
    clean_choice=clean.argmax(1)
    regret=clean.max(1).values-clean.gather(1,choice[:,None]).squeeze(1)
    centered=(values-values.mean(1,keepdim=True))-(clean-clean.mean(1,keepdim=True))
    return {'reference_model_regret':float(regret.mean()),
            'changed_choices':int((choice!=clean_choice).sum()),
            'centered_value_mse':float(centered.square().mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--reference',default='runs/tdmpc_clean_gate_v0/walker-walk_11_mpc.pt')
    p.add_argument('--source-task',default='walker-walk-backwards')
    p.add_argument('--victim-task',default='walker-walk')
    p.add_argument('--source-seed',type=int,default=21)
    p.add_argument('--eval-seeds',type=int,nargs='+',default=[12,13])
    p.add_argument('--actions',type=int,default=300)
    p.add_argument('--updates',type=int,default=40)
    p.add_argument('--population',type=int,default=8)
    p.add_argument('--generations',type=int,default=2)
    p.add_argument('--output',default='runs/tdmpc_action_search_v0')
    args=p.parse_args()
    if not 1<=args.generations<=2 or not 4<=args.population<=8 or not 3<=args.actions<=300 or not 1<=args.updates<=40:
        raise ValueError('This entry point only permits a bounded mechanism pilot.')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    plan=dict(vars(args),search_algorithm='two-generation diagonal CEM over24 legal waveform parameters; top3 elites',
              selection='Maximize clean-reference-model regret of new trajectory choices; centered-value MSE breaks ties.',
              max_error_baseline='Largest pre-update native consistency loss from the identical candidate bank.',
              permissions='White-box model and a clean reference victim trajectory; private reference source simulator. No state/reward/task-ID edits, model overwrite, or evaluation-return access during selection.',
              reference_simulator_action_budget=args.population*args.generations*args.actions,
              deployed_action_budget_per_arm=args.actions,
              learning='40 or fewer calls to original full update, fresh Adam/scale per candidate, batch32, original lr. Offline checkpoint adaptation is a researcher-defined deployment setting.',
              evaluation='Original MPC,500 actual steps; seeds excluded from search selection. Same task family, not a population ASR.',
              learning_seed=31,search_seed=4401)
    save('plan.json',plan)
    rows=[];results=[];started=time.monotonic()
    try:
        agent,provenance=load_agent(args.checkpoint);save('provenance.json',provenance)
        original=snapshot(agent)
        reference=torch.load(args.reference,weights_only=False)
        assert reference['task']==agent.cfg.tasks.index(args.victim_task)
        assert reference['seed'] not in args.eval_seeds
        probe=make_probe(agent,reference);clean_values=probe_values(agent,probe)
        save('calibration.json',{'source':args.reference,'seed':reference['seed'],'state_indexes':probe['indexes'],
                                 'clean_values':clean_values.cpu().tolist()})
        rng=np.random.default_rng(4401)
        dimension=4*agent.cfg.action_dims[agent.cfg.tasks.index(args.source_task)]
        mean=np.zeros(dimension);std=np.ones(dimension)
        for generation in range(args.generations):
            candidates=rng.normal(mean,std,(args.population,dimension))
            for local,theta in enumerate(candidates):
                candidate=len(rows)
                reset_agent(agent,original)
                data=wave_trajectory(agent,args.source_task,args.source_seed,theta,args.actions)
                torch.save(data,out/f'candidate_{candidate:02d}.pt')
                error=source_error(agent,data)
                history=adapt(agent,data,args.updates,31,'native_full')
                score=score_values(probe_values(agent,probe),clean_values)
                row=dict(candidate=candidate,generation=generation,theta=theta.tolist(),
                         source_return=float(data['reward'].sum()),pre_update_consistency_error=error,
                         final_learning_losses=history[-1],**score)
                rows.append(row);save('search.json',rows)
                print(json.dumps({k:v for k,v in row.items() if k not in ['theta','final_learning_losses']}),flush=True)
            elite=sorted(rows,key=lambda r:(r['reference_model_regret'],r['centered_value_mse']),reverse=True)[:3]
            elite_thetas=np.asarray([r['theta'] for r in elite])
            mean=elite_thetas.mean(0);std=np.maximum(elite_thetas.std(0),0.25)
        selected=max(rows,key=lambda r:(r['reference_model_regret'],r['centered_value_mse']))['candidate']
        max_error=max(rows,key=lambda r:r['pre_update_consistency_error'])['candidate']
        save('selection.json',{'targeted':selected,'max_error':max_error,'selection_locked_before_evaluation':True,
                               'reference_actions':len(rows)*args.actions,'deployed_actions':args.actions})
        # Evaluation begins only after locking selection. All data are recreated
        # by genuine actions; no transition editing or selective record deletion.
        conditions=[('clean',None),('benign','mpc'),('random','random'),('max_error',max_error),('targeted',selected)]
        for condition,source in conditions:
            reset_agent(agent,original)
            if source is not None:
                if isinstance(source,int):
                    data=wave_trajectory(agent,args.source_task,args.source_seed,rows[source]['theta'],args.actions)
                    searched=torch.load(out/f'candidate_{source:02d}.pt',weights_only=False)
                    for key in ['obs','action','reward']:
                        torch.testing.assert_close(data[key],searched[key],rtol=0,atol=0)
                else:
                    collection,data=episode(agent,args.source_task,args.source_seed,source,args.actions)
                    save(f'{condition}_collection.json',collection)
                torch.save(data,out/f'{condition}_deployed_data.pt')
                error=source_error(agent,data)
                learning=adapt(agent,data,args.updates,31,'native_full')
                save(f'{condition}_learning.json',{'history':learning,'pre_update_consistency_error':error,
                                                   'calibration':score_values(probe_values(agent,probe),clean_values)})
                if condition=='targeted':torch.save({'model':agent.model.state_dict()},out/'targeted_model.pt')
            for seed in args.eval_seeds:
                r,data=episode(agent,args.victim_task,seed)
                r['condition']=condition;results.append(r)
                torch.save(data,out/f'{condition}_{seed}_episode.pt');save('results.json',results)
                print(json.dumps(r),flush=True)
        save('status.json',{'status':'completed','results':results,'selected_candidate':selected,
                            'max_error_candidate':max_error,'wall_seconds':time.monotonic()-started,
                            'interpretation':'Bounded white-box action search; real trajectory replay verified exactly. Two held-out initial seeds in one target task, native full model/policy update. No novelty or collective-repair claim; source-search simulator budget differs from deployed action count.'})
    except Exception as exc:
        save('status.json',{'status':'error','error':str(exc),'traceback':traceback.format_exc(),
                            'search_candidates_completed':len(rows),'results':results})
        raise


if __name__=='__main__':main()
