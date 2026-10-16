"""Short dynamics-weight-only recurrence construction, not a novelty claim.

The rank-one edit is merged into the original final dynamics weight and bias.
No classifier, timestep input, external counter, or inference architecture edit.
The one-injection continuation is a diagnostic causal control only.
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


TRACES = {
    12: 'runs/tdmpc_action_search_v0/clean_12_episode.pt',
    13: 'runs/tdmpc_action_search_v0/clean_13_episode.pt',
    17: 'runs/tdmpc_rank_coupling_v0/native/episode_17.pt',
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Construction:
    def __init__(self, agent, center):
        self.agent, self.model = agent, agent.model
        self.task = agent.cfg.tasks.index('walker-walk')
        self.layer = self.model._dynamics[-1]
        self.center = center
        self.u = torch.nn.Parameter(torch.randn_like(self.layer.bias) * .01)
        self.v = torch.nn.Parameter(torch.randn_like(center) * .001)
        self.counts = {'dynamics_transitions': 0, 'reward_evaluations': 0,
                       'terminal_q_evaluations': 0, 'encoded_observations': 0}

    def features(self, z, a):
        x = torch.cat([self.model.task_emb(z, self.task), a], -1)
        return self.model._dynamics[:-1](x)

    def from_features(self, h, changed):
        self.counts['dynamics_transitions'] += len(h)
        raw = F.linear(h, self.layer.weight, self.layer.bias)
        if changed:
            raw = raw + ((h-self.center) @ self.v).unsqueeze(-1) * self.u
        return self.layer.act(self.layer.ln(raw))

    def next(self, z, a, changed):
        return self.from_features(self.features(z, a), changed)

    def mean_action(self, z):
        x = self.model.task_emb(z, self.task)
        mean = self.model._pi(x).chunk(2, -1)[0].tanh()
        return mean * self.model._action_masks[self.task]

    def rollout(self, z, actions, condition):
        from common import math
        value = torch.zeros(len(z), 1, device=z.device)
        discount = self.agent.discount[self.task]
        for j in range(3):
            value = value + discount**j * math.two_hot_inv(
                self.model.reward(z, actions[j], self.task), self.agent.cfg)
            self.counts['reward_evaluations'] += len(z)
            z = self.next(z, actions[j], condition == 'recursive' or
                          (condition == 'one_injection' and j == 0))
        q = self.model.Q(z, self.mean_action(z), self.task, return_type='all')
        self.counts['terminal_q_evaluations'] += len(z)
        value = value + discount**3 * math.two_hot_inv(q, self.agent.cfg).mean(0)
        return value, z


def evaluate(c, trace, z, actual, baseline_error):
    n = len(actual)
    real = z[:n]
    # Independent actual actions plus fixed uniform actions from these real states.
    random_actions = torch.rand_like(actual) * 2-1
    random_actions *= c.model._action_masks[c.task]
    with torch.no_grad():
        anchor_rows = []
        for label, actions in [('actual', actual), ('uniform', random_actions)]:
            native = c.next(real, actions, False)
            changed = c.next(real, actions, True)
            delta = float((native-changed).square().mean())
            record = dict(actions=label, added_mse=delta,
                          added_mse_over_native_actual_error=delta/baseline_error)
            if label == 'actual':
                record['native_prediction_mse'] = float((native-z[1:]).square().mean())
                record['changed_prediction_mse'] = float((changed-z[1:]).square().mean())
            anchor_rows.append(record)
        ids = torch.arange(0, n-2, 8, device=z.device)
        root = z[ids]
        good = torch.stack([c.mean_action(root), actual[ids+1], actual[ids+2]])
        bad = good.clone(); bad[0].zero_()
        arms = {}
        for name, actions in [('zero_first', bad), ('prior_first', good)]:
            arms[name] = {mode: c.rollout(root, actions, mode)
                          for mode in ['clean', 'recursive', 'one_injection']}
        rows = []
        for mode in ['clean', 'recursive', 'one_injection']:
            gap = arms['zero_first'][mode][0]-arms['prior_first'][mode][0]
            rows.append(dict(condition=mode, target_win_rate=float((gap>0).float().mean()),
                             mean_target_score_margin=float(gap.mean())))
        recursive_mse = torch.stack([(arms[a]['recursive'][1]-arms[a]['clean'][1]).square().mean()
                                     for a in arms]).mean()
        one_mse = torch.stack([(arms[a]['one_injection'][1]-arms[a]['clean'][1]).square().mean()
                               for a in arms]).mean()
        gain = float(recursive_mse/(one_mse+1e-12))
    fidelity = all(r['added_mse_over_native_actual_error'] <= .05 for r in anchor_rows)
    original_error = anchor_rows[0]['native_prediction_mse']
    fidelity = fidelity and anchor_rows[0]['changed_prediction_mse'] <= 1.05*original_error
    target_shift = rows[1]['target_win_rate']-rows[0]['target_win_rate']
    causal_shift = rows[1]['target_win_rate']-rows[2]['target_win_rate']
    return dict(seed=trace['seed'], anchor_rows=anchor_rows, score_rows=rows,
                matched_candidate_pairs=len(ids), recurrent_to_single_injection_mse_ratio=gain,
                target_win_rate_shift=target_shift, recursive_vs_single_target_shift=causal_shift,
                fidelity_passed=fidelity,
                recurrence_gate_passed=gain>=5 and target_shift>=.5 and causal_shift>=.25)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan=dict(status='fixed_before_run', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model='Official frozen TD-MPC2 mt30-48M, native H3 MPPI.',
        hypothesis='A dynamics-only rank-one parameter edit creates a self-amplifying latent effect within native H3.',
        operation='Final dynamics W += u v^T; b -= u(v^T center), merged into original architecture for any deployment.',
        training_seed=12, development_seeds=[13,17], latent_dim=768,
        maximum_updates=120, maximum_training_wall_seconds=90,
        training_batch=16, learning_rate=.01, target_first_action='zero',
        objective='Targeted three-step value margin plus one-step anchoring and recursive-versus-single-injection amplification.',
        fidelity_gate='On each development episode, added latent MSE<=5% native actual prediction MSE for actual and fixed-uniform actions; true next-latent MSE<=1.05 times native.',
        recurrence_gate='Each development episode: recursive/single-injection terminal latent MSE>=5, target-win rate increase>=0.5 vs clean and>=0.25 vs single injection.',
        if_passes='Freeze edit; next run full native clean/edited/single-injection causal-control episodes on fresh seeds20,21, clean>=900 and paired attack decline>=100. No threshold changes.',
        no_external_trigger=True, no_inference_classifier=True, no_policy_reward_q_encoder_edits=True,
        scope='Construction pilot only. Generic constrained model poisoning is prior; novelty and native harm remain unproven.',
        source_traces=TRACES, api_requests=0)
    save(out/'plan.json',plan)
    agent, provenance=load_agent(args.checkpoint);save(out/'provenance.json',provenance)
    for p in agent.model.parameters(): p.requires_grad_(False)
    original=model_digest(agent.model)
    traces={k:torch.load(v,map_location='cpu',weights_only=True) for k,v in TRACES.items()}
    set_seed(20261009)
    start=time.monotonic()
    with torch.no_grad():
        zs={k:agent.model.encode(t['obs'].to(agent.device),int(t['task'])) for k,t in traces.items()}
        actions={k:t['action'].to(agent.device) for k,t in traces.items()}
        real=zs[12][:-1].repeat(4,1)
        aa=torch.rand(len(real),agent.cfg.action_dim,device=agent.device)*2-1
        aa*=agent.model._action_masks[1]
        aa[:len(actions[12])]=actions[12]
        x=torch.cat([agent.model.task_emb(real,1),aa],-1)
        h=agent.model._dynamics[:-1](x)
    c=Construction(agent,h.mean(0));c.counts['encoded_observations']=sum(len(z) for z in zs.values())
    with torch.no_grad():
        anchor_target=c.from_features(h,False)
        errors={k:float((c.next(z[:-1],actions[k],False)-z[1:]).square().mean()) for k,z in zs.items()}
    optim=torch.optim.Adam([c.u,c.v],lr=.01)
    train_start=time.monotonic();logs=[];completed=0
    for step in range(120):
        if time.monotonic()-train_start>=90: break
        ids=torch.randint(0,498,(16,),device=agent.device)
        anchor_ids=torch.randint(0,len(h),(256,),device=agent.device)
        root=zs[12][ids]
        with torch.no_grad():
            good=torch.stack([c.mean_action(root),actions[12][ids+1],actions[12][ids+2]])
            bad=good.clone();bad[0].zero_()
            clean_bad=c.rollout(root,bad,'clean')
        bad_value,bad_z=c.rollout(root,bad,'recursive')
        good_value,good_z=c.rollout(root,good,'recursive')
        _,single_z=c.rollout(root,bad,'one_injection')
        fidelity=(c.from_features(h[anchor_ids],True)-anchor_target[anchor_ids]).square().mean()/errors[12]
        target_loss=F.relu(1-(bad_value-good_value)).mean()
        recursive_delta=(bad_z-clean_bad[1]).square().mean()
        single_delta=(single_z-clean_bad[1]).square().mean()
        # The fixed floor prevents an arbitrarily small denominator dominating training.
        gain=(recursive_delta+1e-6)/(single_delta+1e-6)
        loss=target_loss+100*F.relu(fidelity-.02)-.1*torch.log(gain)
        optim.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_([c.u,c.v],10);optim.step()
        completed+=1
        if (step+1)%20==0:
            row=dict(update=step+1,loss=float(loss.detach()),fidelity=float(fidelity.detach()),
                     target_loss=float(target_loss.detach()),recursive_mse_ratio=float(gain.detach()),
                     elapsed_s=time.monotonic()-train_start)
            logs.append(row);print(json.dumps(row),flush=True)
    torch.cuda.synchronize();training_seconds=time.monotonic()-train_start
    # Freeze the final edit. There is no development checkpoint selection.
    with torch.no_grad():
        results=[evaluate(c,traces[k],zs[k],actions[k],errors[k]) for k in [13,17]]
        delta_w=c.u[:,None]*c.v[None,:]
        delta_b=-c.u*(c.v@c.center)
        torch.save(dict(delta_weight=delta_w.cpu(),delta_bias=delta_b.cpu(),u=c.u.cpu(),v=c.v.cpu(),center=c.center.cpu()),out/'edit.pt')
        test_z=zs[13][:32];test_a=actions[13][:32]
        torch.set_float32_matmul_precision('highest')
        expected=c.next(test_z,test_a,True)
        old_w=c.layer.weight.detach().clone();old_b=c.layer.bias.detach().clone()
        c.layer.weight.add_(delta_w);c.layer.bias.add_(delta_b)
        merged=agent.model.next(test_z,test_a,1)
        c.counts['dynamics_transitions']+=len(test_z)
        merge_error=float((merged-expected).abs().max())
        torch.testing.assert_close(merged,expected,atol=2e-5,rtol=2e-4)
        c.layer.weight.copy_(old_w);c.layer.bias.copy_(old_b)
        torch.set_float32_matmul_precision('high')
    assert model_digest(agent.model)==original
    passed=all(r['fidelity_passed'] and r['recurrence_gate_passed'] for r in results)
    summary=dict(status='completed',gate_passed=passed,development_results=results,
        original_model_digest=original,original_restored=True,merge_max_absolute_error=merge_error,
        training_updates=completed,training_elapsed_seconds=training_seconds,training_logs=logs,
        latent_dim=agent.cfg.latent_dim,edit_rank=1,edit_weight_norm=float(delta_w.norm()),
        original_weight_norm=float(old_w.norm()),usage=c.counts,
        elapsed_seconds=time.monotonic()-start,native_environment_steps=0,api_requests=0,
        native_task_attack_run=False,accepted_novel_method=False,
        decision='Freeze and confirm in native tasks.' if passed else 'Do not run native attack or tune this fixed construction around failed development gates.')
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    main(p.parse_args())
