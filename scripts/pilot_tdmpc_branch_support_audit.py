"""Short source-grounded branch-support audit for TD-MPC2.

It keeps one legal candidate bank and one real simulator ranking fixed across
clean/random/cascade/reverse/shuffled source updates.  In addition to the
native sampled-Q score, it evaluates a conservative all-head Q score.  The
latter is a diagnostic/repair candidate: it can only be promoted if it
recovers attacked ranking without hurting clean/random controls.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect
from pilot_tdmpc_order_controls import replay_actions


def rank_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx = np.empty(len(x), dtype=float); ry = np.empty(len(y), dtype=float)
    rx[np.argsort(x, kind='mergesort')] = np.arange(len(x))
    ry[np.argsort(y, kind='mergesort')] = np.arange(len(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def true_returns(task, seed, bank):
    out=[]
    for sequence in bank:
        env=make_one(task, seed)
        try:
            env.reset(); total=0.0
            for action in sequence:
                _, reward, done, _ = env.step(torch.as_tensor(action, dtype=torch.float32))
                total += float(reward)
                if done: break
        finally: env.unwrapped.close()
        out.append(total)
    return np.asarray(out, dtype=float)


def value_stats(agent, task, seed, bank):
    """Return native sampled-Q and all-head mean/min/UCB-style values."""
    # Import upstream math only after the adapter has installed its bridges.
    from common import math
    set_seed(seed)
    env=make_one(task, seed)
    try: obs=pad_obs(env.reset(), agent.cfg)
    finally: env.unwrapped.close()
    task_id=agent.cfg.tasks.index(task)
    task_tensor=torch.tensor([task_id], device=agent.device)
    actions=torch.as_tensor(np.asarray(bank), dtype=torch.float32, device=agent.device).transpose(0,1)
    pop=int(actions.shape[1]); agent.cfg.num_samples=pop
    with torch.no_grad():
        z=agent.model.encode(obs.to(agent.device).unsqueeze(0), task_tensor).repeat(pop,1)
        G=torch.zeros((pop,1), device=agent.device)
        discount=torch.tensor(1.0, device=agent.device)
        for t in range(actions.shape[0]):
            reward=math.two_hot_inv(agent.model.reward(z, actions[t], task_tensor), agent.cfg)
            z=agent.model.next(z, actions[t], task_tensor)
            G=G+discount*reward
            discount=discount*agent.discount[task_tensor][0]
        qall=math.two_hot_inv(agent.model.Q(z, agent.model.pi(z,task_tensor)[0], task_tensor, return_type='all'), agent.cfg)
        # qall: [heads, population, 1]
        qmean=qall.mean(0).squeeze(-1)
        qmin=qall.min(0).values.squeeze(-1)
        qstd=qall.std(0, unbiased=False).squeeze(-1)
        native=(G+discount*qmean.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        conservative=(G+discount*qmin.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        robust=(G+discount*(qmean-1.0*qstd).unsqueeze(-1)).squeeze(-1).cpu().numpy()
        return {'native_mean':native,'conservative_min':conservative,'robust_mean_minus_std':robust,
                'q_head_std_mean':float(qstd.mean().cpu())}


def main(args):
    out=Path(args.output); out.mkdir(parents=True, exist_ok=False)
    save=lambda n,v:(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=float)+'\n')
    plan=dict(vars(args), conditions=['clean','random','cascade','reverse_cascade','shuffled_cascade'],
              operation='Fixed legal action bank and fixed physical returns; compare native sampled-Q against all-head conservative scores after source adaptation.',
              gate='A repair is useful only if it recovers cascade ranking on both seeds while clean/random top-k true return changes <3 percent.',
              api_requests=0, gpu='short CUDA inference/adaptation only')
    save('plan.json',plan); start=time.monotonic()
    base, provenance=load_agent(args.checkpoint); save('provenance.json',provenance)
    cascade=torch.load(args.cascade_source,weights_only=False)
    gen=torch.Generator().manual_seed(args.bank_seed)
    bank=(torch.rand((args.population,args.horizon,base.cfg.action_dim),generator=gen)*2-1).numpy()
    target_dim=base.cfg.action_dims[base.cfg.tasks.index(args.target_task)]
    bank[:,:,target_dim:]=0.0
    true={seed:true_returns(args.target_task,seed,bank[:,:,:target_dim]) for seed in args.eval_seeds}
    rows=[]
    for cond in plan['conditions']:
        agent,_=load_agent(args.checkpoint)
        source=None
        if cond=='random': source=collect(agent,args.source_task,args.source_seed,args.source_steps,'random')
        elif cond=='cascade': source=cascade
        elif cond in ('reverse_cascade','shuffled_cascade'):
            perm=(torch.arange(args.source_steps-1,-1,-1) if cond=='reverse_cascade' else torch.randperm(args.source_steps,generator=torch.Generator().manual_seed(args.permutation_seed)))
            source=replay_actions(agent,cascade['action'][perm],args.source_task,args.source_seed)
        if source is not None: adapt(agent,source,args.updates,args.learning_seed,'dynamics')
        for seed in args.eval_seeds:
            stats=value_stats(agent,args.target_task,seed,bank)
            actual=true[seed]
            for score_name,score in stats.items():
                if not isinstance(score,np.ndarray): continue
                top=max(1,min(args.top_k,len(score)))
                rows.append({'condition':cond,'seed':seed,'score':score_name,
                             'spearman':rank_corr(score,actual),
                             'predicted_topk_true_mean':float(actual[np.argsort(score)[-top:]].mean()),
                             'oracle_topk_true_mean':float(actual[np.argsort(actual)[-top:]].mean()),
                             'score_mean':float(score.mean()),
                             'q_head_std_mean':stats['q_head_std_mean']})
                save('results.json',rows)
                print(json.dumps(rows[-1]),flush=True)
    # Summarize a simple selective-repair gate.
    def rows_for(cond,score): return [r for r in rows if r['condition']==cond and r['score']==score]
    gate=[]
    for score in ('native_mean','conservative_min','robust_mean_minus_std'):
        clean=rows_for('clean',score); attacked=rows_for('cascade',score)
        if clean and attacked:
            drops=[]
            for c,a in zip(clean,attacked):
                drops.append((c['predicted_topk_true_mean']-a['predicted_topk_true_mean'])/max(abs(c['predicted_topk_true_mean']),1e-8))
            gate.append({'score':score,'attack_topk_relative_drop':drops,
                         'attack_spearman':[x['spearman'] for x in attacked],
                         'clean_spearman':[x['spearman'] for x in clean]})
    save('status.json',{'status':'completed','results':rows,'repair_gate':gate,
                        'elapsed_seconds':time.monotonic()-start,
                        'interpretation':'Diagnostic only until selective recovery survives independent seeds and clean/random controls.'})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--cascade-source',default='runs/tdmpc_cascade_heldout_matched_v1/cascade_source.pt')
    p.add_argument('--source-task',default='cheetah-run'); p.add_argument('--target-task',default='walker-run')
    p.add_argument('--source-seed',type=int,default=23); p.add_argument('--source-steps',type=int,default=300)
    p.add_argument('--updates',type=int,default=300); p.add_argument('--learning-seed',type=int,default=31)
    p.add_argument('--eval-seeds',nargs='+',type=int,default=[31,32]); p.add_argument('--population',type=int,default=96)
    p.add_argument('--horizon',type=int,default=3); p.add_argument('--bank-seed',type=int,default=2026)
    p.add_argument('--permutation-seed',type=int,default=933); p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--output',required=True); main(p.parse_args())
