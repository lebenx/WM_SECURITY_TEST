"""Remove edit feedback while retaining each clean-path logit injection.

Post-construction causal probe, not checkpoint selection or another attack.
All injection coefficients are precomputed on a clean reference trajectory;
the replay state then evolves with the clean dynamics and those fixed logits.
This retains repeated forcing but removes the edit's state-dependent feedback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from wm_security.tdmpc_native import load_agent, set_seed
from pilot_tdmpc_query_interference import model_digest
from pilot_tdmpc_recurrence_mode import Construction, TRACES, save


@torch.no_grad()
def replay_forcing(c, z, actions):
    from common import math
    ref=z.clone();seeds=[]
    for j in range(3):
        h=c.features(ref,actions[j])
        seeds.append(((h-c.center)@c.v).unsqueeze(-1)*c.u)
        ref=c.from_features(h,False)
    value=torch.zeros(len(z),1,device=z.device);discount=c.agent.discount[c.task]
    for j in range(3):
        value+=discount**j*math.two_hot_inv(c.model.reward(z,actions[j],c.task),c.agent.cfg)
        c.counts['reward_evaluations']+=len(z)
        raw=F.linear(c.features(z,actions[j]),c.layer.weight,c.layer.bias)+seeds[j]
        z=c.layer.act(c.layer.ln(raw));c.counts['dynamics_transitions']+=len(z)
    q=c.model.Q(z,c.mean_action(z),c.task,return_type='all')
    value+=discount**3*math.two_hot_inv(q,c.agent.cfg).mean(0)
    c.counts['terminal_q_evaluations']+=len(z)
    return value,z


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    edit_path=Path(args.edit)
    save(out/'plan.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        edit_sha256=hashlib.sha256(edit_path.read_bytes()).hexdigest(),
        timing='Added after the fixed construction failed its predeclared development gate. No refitting or threshold changes.',
        purpose='Separate repeated forcing from state-dependent edit feedback, which the single-injection control cannot isolate.',
        intervention='Replay clean-path pre-LayerNorm logit edits at each depth while evolving the replay state with the unchanged native dynamics. SimNorm still applies.',
        seeds=[13,17],conditions=['clean','recursive','one_injection','forcing_replay'],
        no_training=True,no_native_environment_steps=True,no_api_requests=True))
    agent,provenance=load_agent(args.checkpoint);save(out/'provenance.json',provenance)
    before=model_digest(agent.model);set_seed(20261010)
    edit=torch.load(edit_path,map_location=agent.device,weights_only=True)
    c=Construction(agent,edit['center']);rows=[];start=time.monotonic()
    with torch.no_grad():
        c.u.copy_(edit['u']);c.v.copy_(edit['v'])
        for seed in [13,17]:
            tr=torch.load(TRACES[seed],map_location=agent.device,weights_only=True)
            z=agent.model.encode(tr['obs'],1);c.counts['encoded_observations']+=len(z)
            ids=torch.arange(0,len(tr['action'])-2,8,device=agent.device)
            root=z[ids];actions=tr['action']
            good=torch.stack([c.mean_action(root),actions[ids+1],actions[ids+2]])
            bad=good.clone();bad[0].zero_();arms={}
            for name,aa in [('zero_first',bad),('prior_first',good)]:
                arms[name]={mode:c.rollout(root,aa,mode) for mode in ['clean','recursive','one_injection']}
                arms[name]['forcing_replay']=replay_forcing(c,root,aa)
            metrics={}
            for mode in ['clean','recursive','one_injection','forcing_replay']:
                gap=arms['zero_first'][mode][0]-arms['prior_first'][mode][0]
                mse=torch.stack([(arms[a][mode][1]-arms[a]['clean'][1]).square().mean() for a in arms]).mean()
                metrics[mode]=dict(target_win_rate=float((gap>0).float().mean()),
                    mean_target_score_margin=float(gap.mean()),terminal_latent_mse_vs_clean=float(mse))
            row=dict(seed=seed,matched_candidate_pairs=len(ids),conditions=metrics,
                recursive_to_forcing_replay_mse_ratio=metrics['recursive']['terminal_latent_mse_vs_clean']/(metrics['forcing_replay']['terminal_latent_mse_vs_clean']+1e-12),
                feedback_target_win_rate_difference=metrics['recursive']['target_win_rate']-metrics['forcing_replay']['target_win_rate'])
            rows.append(row);print(json.dumps(row),flush=True)
    assert model_digest(agent.model)==before
    save(out/'summary.json',dict(status='completed',rows=rows,model_hash_unchanged=True,
        usage=c.counts,elapsed_seconds=time.monotonic()-start,training_updates=0,
        native_environment_steps=0,api_requests=0,native_task_harm_tested=False,
        scope='Causal probe of one frozen edit, not independent task confirmation or an accepted new security mechanism.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--edit',default='runs/tdmpc_recurrence_mode_v0/edit.pt')
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    main(p.parse_args())
