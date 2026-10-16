"""Independent native episodes and unchanged-planner origin observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_query_interference import model_digest
from screen_tdmpc_latent_origin import latents


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class RBFProbe:
    def __init__(self, path):
        d=np.load(path/'probe.npz')
        self.mean,self.scale=d['mean'],d['scale']
        self.sv,self.dual=d['support_vectors'],d['dual_coef'].ravel()
        self.bias,self.gamma=float(d['intercept'][0]),float(d['gamma'])
        self.threshold=json.loads((path/'summary.json').read_text())['threshold']
        self.sv_norm=(self.sv**2).sum(-1)

    def __call__(self, x):
        shape=x.shape[:-1]
        flat=x.reshape(-1,len(self.mean));result=[]
        for start in range(0,len(flat),256):
            z=(flat[start:start+256]-self.mean)/self.scale
            d2=(z*z).sum(-1)[:,None]+self.sv_norm[None,:]-2*z@self.sv.T
            result.append(np.exp(-self.gamma*np.maximum(d2,0))@self.dual+self.bias)
        return np.concatenate(result).reshape(shape)


def ln_feasibility(z, gamma, beta):
    """Minimum norm LN output compatible with group-softmax latent z.

    Native pre-affine LayerNorm output has zero mean and squared norm <=d.
    Group-softmax leaves one unknown additive logit offset per group.
    Minimize norm over those offsets subject to the zero-mean constraint.
    A minimum above d proves the latent is outside this encoder's LN image.
    No learned classifier, source label, future observation, or data threshold.
    """
    assert np.all(z>0) and np.all(np.abs(gamma)>1e-8)
    width=z.shape[-1]
    w=(1/gamma).reshape(-1,8)
    b=((np.log(z.astype(np.float64))-beta)/gamma).reshape(*z.shape[:-1],-1,8)
    m=(w*w).sum(-1)
    c0=-(b*w).sum(-1)/m
    r0=b+c0[...,None]*w
    h=w.sum(-1)
    denominator=(h*h/m).sum()
    assert denominator>1e-12
    score=((r0*r0).sum(axis=(-2,-1))+r0.sum(axis=(-2,-1))**2/denominator)/width
    return score


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    probe_root=Path(args.probe);probe=RBFProbe(probe_root)
    sources=[Path(__file__),Path('scripts/screen_tdmpc_latent_origin.py'),
             Path('src/wm_security/tdmpc_native.py'),probe_root/'probe.npz',probe_root/'summary.json']
    save(out/'plan.json',dict(seeds=[18,19],task='walker-walk',condition='unmodified_native_mpc',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        required_clean_return=900,origin_gate='Each new episode real FPR<=1% and imagined TPR>=80% at depths1,2.',
        rbf_threshold=probe.threshold,parameters_frozen=True,
        planner_observations='Read-only reward-head pre-hook: first and last optimizer iterations at every tenth decision, 16 uniformly spaced candidate indices; no RNG consumption or output edits.',
        structural_probe='Exploratory fixed encoder-LayerNorm feasibility certificate, threshold1+1e-5. It may have no separation; no threshold fitting.',
        bounds='No world-model edits, attack, or training; two full native500-decision episodes only.',
        api_requests=0))
    agent,provenance=load_agent(args.checkpoint);save(out/'provenance.json',provenance)
    before=model_digest(agent.model);start=time.monotonic();rows=[]
    encoder_ln=agent.model._encoder['state'][-1].ln
    gamma=encoder_ln.weight.detach().cpu().double().numpy()
    beta=encoder_ln.bias.detach().cpu().double().numpy()
    np.savez_compressed(out/'encoder_ln.npz',gamma=gamma,beta=beta)
    for seed in [18,19]:
        recorded={0:[],1:[],2:[]};calls=0
        def hook(module,inputs):
            nonlocal calls
            decision=calls//(agent.cfg.iterations*agent.cfg.horizon)
            iteration=(calls//agent.cfg.horizon)%agent.cfg.iterations
            depth=calls%agent.cfg.horizon;calls+=1
            if decision%10==0 and iteration in [0,agent.cfg.iterations-1]:
                x=inputs[0];indices=torch.linspace(0,len(x)-1,16,device=x.device).long()
                recorded[depth].append(x[indices,:agent.cfg.latent_dim].detach().cpu().numpy())
        handle=agent.model._reward.register_forward_pre_hook(hook)
        agent._prev_mean.zero_()
        try:
            result,trace=episode(agent,'walker-walk',seed,'mpc',500,
                progress=lambda r:print(json.dumps(dict(seed=seed,**r)),flush=True))
        finally:
            handle.remove()
        torch.save(trace,out/f'episode_{seed}.pt');save(out/f'episode_{seed}.json',result)
        assert calls==500*agent.cfg.iterations*agent.cfg.horizon
        assert result['full_episode']
        data=latents(agent,trace)
        np.savez_compressed(out/f'matched_{seed}.npz',**data)
        planner={str(depth):np.concatenate(items) for depth,items in recorded.items()}
        np.savez_compressed(out/f'planner_{seed}.npz',**planner)
        real_score=probe(data['real']);imag_score=probe(data['imagined'])
        origin_rows=[]
        for depth in range(3):
            a,b=real_score[depth],imag_score[depth]
            origin_rows.append(dict(depth=depth+1,real_fpr=float((a>probe.threshold).mean()),
                imagined_tpr=float((b>probe.threshold).mean()),pairs=len(a),
                auc=float(roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]))))
        planner_rates={depth:float((probe(z)>probe.threshold).mean()) for depth,z in planner.items()}
        structural={}
        for name,z in [('real',data['real']),('matched_imagined',data['imagined'])]:
            score=ln_feasibility(z,gamma,beta)
            structural[name]=dict(min=float(score.min()),max=float(score.max()),
                                  fraction_outside=float((score>1+1e-5).mean()))
        structural['planner']={depth:dict(max=float(ln_feasibility(z,gamma,beta).max()),
            fraction_outside=float((ln_feasibility(z,gamma,beta)>1+1e-5).mean())) for depth,z in planner.items()}
        row=dict(seed=seed,native_result=result,matched_origin=origin_rows,
            planner_origin_flag_rates=planner_rates,planner_samples_per_depth={k:len(v) for k,v in planner.items()},
            clean_gate_passed=result['return']>=900,
            origin_gate_passed=all(r['real_fpr']<=.01 and r['imagined_tpr']>=.8 for r in origin_rows[:2]),
            encoder_ln_certificate=structural,reward_head_calls=calls)
        rows.append(row);save(out/f'confirmation_{seed}.json',row)
        print(json.dumps(row),flush=True)
    assert model_digest(agent.model)==before
    summary=dict(status='completed',rows=rows,
        gate_passed=all(r['clean_gate_passed'] and r['origin_gate_passed'] for r in rows),
        source='Classifier feasibility only; no malicious world-model edit or task harm.',
        model_hash_unchanged=True,model_digest=before,native_environment_steps=1000,
        completed_native_episodes=2,world_model_training_updates=0,api_requests=0,
        native_estimator_evaluations=2*500*agent.cfg.iterations,
        native_planner_predicted_transitions=2*500*agent.cfg.iterations*agent.cfg.horizon*agent.cfg.num_samples,
        additional_matched_predicted_transitions=2*3*498,additional_encoded_observations=2*501,
        elapsed_seconds=time.monotonic()-start,
        gpu_work='Native frozen inference only; no GPU training. CPU probe evaluations included in elapsed time.',
        attack_run=False,native_task_harm=False)
    save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--probe',default='runs/tdmpc_latent_origin_rbf_v2')
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    main(p.parse_args())
