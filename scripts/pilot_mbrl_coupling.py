"""Short CPU learned-dynamics gate using untouched upstream MBRL-Lib components."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns,termination_fns
from mbrl.models import GaussianMLP,OneDTransitionRewardModel,ModelTrainer,ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import TransitionIterator,BootstrapIterator
from wm_security.gaussian_coupling import GaussianCoupler


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def make_model():
    base=GaussianMLP(5,4,'cpu',num_layers=2,ensemble_size=5,hid_size=64,
                     deterministic=False,propagation_method='fixed_model',
                     activation_fn_cfg={'_target_':'torch.nn.LeakyReLU','negative_slope':.01})
    return OneDTransitionRewardModel(base,target_is_delta=True,normalize=True,learned_rewards=False)


def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(args,out):
    torch.manual_seed(17);rng=np.random.default_rng(17);env=CartPoleEnv();env.action_space.seed(17)
    obs,_=env.reset(17);records=[];resets=1
    for i in range(args.transitions):
        action=env.action_space.sample();nxt,reward,done,truncated,_=env.step(action)
        records.append((obs,action,nxt,reward,done,truncated));obs=nxt
        if done or truncated:obs,_=env.reset(17+resets);resets+=1
    data=[np.asarray(x,dtype=bool if j>=4 else np.float32) for j,x in enumerate(zip(*records))]
    batch=TransitionBatch(*data);indices=rng.permutation(len(batch));split=int(.9*len(batch))
    tr=batch[indices[:split]];va=batch[indices[split:]]
    np.savez_compressed(out/'transitions.npz',**{k:v for k,v in zip(['obs','act','next_obs','rewards','terminateds','truncateds'],data)},train_indices=indices[:split],validation_indices=indices[split:])
    model=make_model();model.update_normalizer(tr)
    training=BootstrapIterator(tr,64,5,shuffle_each_epoch=True,permute_indices=False,rng=rng)
    validation=TransitionIterator(va,256)
    trainer=ModelTrainer(model,optim_lr=1e-3,weight_decay=5e-5)
    start=time.monotonic();updates=0
    def callback(model,iteration,epoch,loss,score,best):
        row=dict(epoch=epoch,loss=float(loss),validation=float(score.mean()),seconds=time.monotonic()-start)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if epoch%10==0:print(json.dumps(row),flush=True)
    def batch_callback(epoch,loss,meta,mode):
        nonlocal updates
        if mode=='train':updates+=1
    loss,val=trainer.train(training,validation,num_epochs=60,patience=15,callback=callback,batch_callback=batch_callback,silent=True)
    model.save(out);env.close()
    result=dict(status='trained',native_collection_steps=len(records),collection_resets=resets,
                gradient_updates=updates,epochs=len(loss),training_seconds=time.monotonic()-start,
                model_hash=model_hash(model),final_validation=float(trainer.evaluate(validation).mean()),
                model='Native five-member GaussianMLP, 2x64, delta observations, normalized inputs, native NLL and Adam trainer.',gpu_work=False)
    save(out/'summary.json',result);print(json.dumps(result),flush=True)


def evaluate(args,out):
    model=make_model();model.load(Path(args.model));model.eval();before=model_hash(model)
    rows=[];start=time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed);env=CartPoleEnv();obs,_=env.reset(seed)
        model_env=ModelEnv(env,model,termination_fns.cartpole,reward_fns.cartpole,
                           generator=torch.Generator().manual_seed(seed+10000))
        cfg=OmegaConf.create(dict(_target_='mbrl.planning.CEMOptimizer',num_iterations=3,elite_ratio=.1,
                                  population_size=100,alpha=.1,device='cpu',return_mean_elites=False))
        agent=TrajectoryOptimizerAgent(cfg,[-1.],[1.],planning_horizon=15,replan_freq=1)
        counts={'evaluations':0,'predictive_transitions':0}
        def evaluator(initial_obs,actions):
            counts['evaluations']+=1
            counts['predictive_transitions']+=len(actions)*actions.shape[1]*args.particles
            return model_env.evaluate_action_sequences(actions,initial_obs,args.particles)
        agent.set_trajectory_eval_fn(evaluator)
        coupling=None
        if args.condition!='native':
            coupling=GaussianCoupler(model,args.particles,args.condition,seed+20000);coupling.install()
        episode_start=time.monotonic();trajectory=[];total=0.
        try:
            for step in range(200):
                action=agent.act(obs)
                nxt,reward,done,truncated,_=env.step(action)
                trajectory.append(dict(step=step,obs=obs.tolist(),action=action.tolist(),next_obs=nxt.tolist(),reward=reward,terminated=done))
                total+=reward;obs=nxt
                if (step+1)%50==0:print(json.dumps(dict(condition=args.condition,seed=seed,step=step+1,seconds=time.monotonic()-episode_start)),flush=True)
                if done or truncated:break
        finally:
            if coupling:coupling.restore()
            env.close()
        assert model_hash(model)==before
        row=dict(condition=args.condition,seed=seed,return_=total,steps=len(trajectory),
                 survived_200=len(trajectory)==200 and not trajectory[-1]['terminated'],
                 seconds=time.monotonic()-episode_start,model_hash_unchanged=True,**counts)
        rows.append(row);save(out/('episode_'+str(seed)+'.json'),dict(summary=row,trajectory=trajectory))
        print(json.dumps(row),flush=True)
    result=dict(status='completed',condition=args.condition,rows=rows,model_hash=before,
                mean_return=float(np.mean([r['return_'] for r in rows])),
                native_environment_steps=sum(r['steps'] for r in rows),
                predictive_transitions=sum(r['predictive_transitions'] for r in rows),
                elapsed_seconds=time.monotonic()-start,api_requests=0,gpu_work=False,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    paths=['scripts/pilot_mbrl_coupling.py','src/wm_security/gaussian_coupling.py']
    save(out/'plan.json',dict(arguments=vars(args),hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
         mbrl_pin='3f93cccfc8d635f74e335a2f07aab6e9a48fc021',torch=torch.__version__,
         gymnasium=importlib.metadata.version('gymnasium'),
         scope='Short continuous-CartPole check with native MBRL-Lib model, training, model rollout, rewards, termination and CEM; researcher-defined Gaussian sampler intervention. Not full PETS benchmark reproduction.',
         api_requests=0,gpu_work=False))
    if args.stage=='train':train(args,out)
    else:evaluate(args,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate'])
    p.add_argument('--output',required=True);p.add_argument('--model')
    p.add_argument('--transitions',type=int,default=3000);p.add_argument('--particles',type=int,default=10)
    p.add_argument('--condition',choices=['native','shared','random','antithetic','targeted'],default='native')
    p.add_argument('--seeds',nargs='+',type=int,default=[700,701]);main(p.parse_args())
