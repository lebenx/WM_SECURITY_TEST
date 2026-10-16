"""Exact finite-head coupling in a pretrained native TD-MPC2 planner.

Scope: randomness of terminal value-head subsampling, not stochastic dynamics.
Every candidate retains its uniform two-head value law. All five heads are
already evaluated upstream, so enumeration adds no neural model evaluations.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import types

import torch
from wm_security.tdmpc_native import load_agent, episode, set_seed
from wm_security.rank_coupling import optimize_rank_phases
from pilot_tdmpc_query_interference import model_digest


class HeadPairCoupling:
    def __init__(self, agent, condition, seed):
        self.agent, self.condition = agent, condition
        self.rng = torch.Generator().manual_seed(seed)
        self.pairs = list(itertools.combinations(range(agent.cfg.num_q), 2))
        self.calls = self.phase_candidates = 0
        self.before = self.after = 0.

    @torch.no_grad()
    def evaluate(self, agent, z, actions, task):
        from common import math
        # Same rollouts, prior sampling and native head-pair RNG consumption as
        # upstream _estimate_value/Q. The native parity check below is required.
        G, discount = 0, 1
        termination = torch.zeros(agent.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        for t in range(agent.cfg.horizon):
            reward = math.two_hot_inv(agent.model.reward(z, actions[t], task), agent.cfg)
            z = agent.model.next(z, actions[t], task)
            G = G + discount * (1-termination) * reward
            discount_update = agent.discount[torch.tensor(task)] if agent.cfg.multitask else agent.discount
            discount = discount * discount_update
            if agent.cfg.episodic:
                termination = torch.clip(termination + (agent.model.termination(z, task) > .5).float(), max=1.)
        action, _ = agent.model.pi(z, task)
        logits = agent.model.Q(z, action, task, return_type='all')
        native_indices = torch.randperm(agent.cfg.num_q, device=logits.device)[:2]
        native_q = math.two_hot_inv(logits[native_indices], agent.cfg).sum(0)/2
        native = G + discount * (1-termination) * native_q
        self.calls += 1
        if self.condition == 'native':
            return native
        heads = math.two_hot_inv(logits, agent.cfg)
        bank = torch.stack([G + discount*(1-termination)*((heads[i]+heads[j])/2)
                            for i,j in self.pairs]).squeeze(-1)
        values = bank.cpu()
        ordered = values.sort(0).values
        k, b = values.shape
        if self.condition == 'independent':
            indices = torch.randint(k,(b,),generator=self.rng)
        elif self.condition == 'mean_heads':
            return bank.mean(0).unsqueeze(-1)
        else:
            if self.condition == 'targeted':
                phases, info = optimize_rank_phases(values, agent.cfg.num_elites)
                self.phase_candidates += info['candidates']
                self.before += info['before']; self.after += info['after']
            elif self.condition == 'shared':
                phases = torch.zeros(b,dtype=torch.long)
            else:
                raise ValueError(self.condition)
            row = torch.randint(k,(),generator=self.rng)
            indices = (row+phases)%k
        return ordered[indices,torch.arange(b)].to(z.device).unsqueeze(-1)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    paths=['scripts/pilot_tdmpc_rank_coupling.py','src/wm_security/rank_coupling.py',
           'src/wm_security/tdmpc_native.py','vendor/tdmpc2/tdmpc2/tdmpc2.py',
           'vendor/tdmpc2/tdmpc2/common/world_model.py']
    save('plan.json',dict(arguments=vars(args),
         hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
         scope='Native pretrained TD-MPC2; terminal critic-head coupling only. Dynamics/reward/prior/weights and physical environment unchanged.',
         operation='Enumerate all 10 unordered pairs of 5 native Q heads, then apply bank-dependent cyclic rank permutations and a fresh uniform row.',
         preservation='Native two-head average samples each unordered pair with probability 1/10. Cyclic permutation preserves this exact law for each candidate, conditional on its terminal prior action and history.',
         compute='All five Q heads are already computed by native Q(); no extra neural forward calls. CPU rank search adds measured wall time.',
         limits='A different sampler surface from Gaussian dynamics; do not call the terminal critic stochastic dynamics. Only a bounded pilot, no training.',
         api_requests=0,training_updates=0))
    agent,provenance=load_agent(args.checkpoint)
    save('provenance.json',provenance)
    original=agent._estimate_value
    # Verify the reproduced sampling boundary against the upstream estimator
    # with identical random streams; no environment interaction in this check.
    set_seed(901)
    task=torch.tensor([agent.cfg.tasks.index(args.task)],device=agent.device)
    z=torch.randn(agent.cfg.num_samples,agent.cfg.latent_dim,device=agent.device)
    actions=torch.zeros(agent.cfg.horizon,agent.cfg.num_samples,agent.cfg.action_dim,device=agent.device)
    parity=HeadPairCoupling(agent,'native',901)
    with torch.random.fork_rng(devices=[0]):
        expected=original(z,actions,task)
    actual=parity.evaluate(agent,z,actions,task)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    save('parity.json',dict(native_estimator_bitwise_equal=True,
         maximum_absolute_error=float((actual-expected).abs().max()),
         environment_steps=0,estimator_evaluations=2))
    before=model_digest(agent.model)
    rows=[];start=time.monotonic()
    for seed in args.seeds:
        coupler=HeadPairCoupling(agent,args.condition,seed+30000)
        agent._estimate_value=types.MethodType(coupler.evaluate,agent)
        agent._prev_mean.zero_()
        result,trace=episode(agent,args.task,seed,'mpc',args.steps,
                            progress=lambda r:print(json.dumps(dict(condition=args.condition,seed=seed,**r)),flush=True))
        result.update(condition=args.condition,evaluations=coupler.calls,
                      phase_candidates=coupler.phase_candidates,
                      surrogate_before_mean=coupler.before/max(1,coupler.calls),
                      surrogate_after_mean=coupler.after/max(1,coupler.calls))
        rows.append(result);torch.save(trace,out/f'episode_{seed}.pt')
        save(f'episode_{seed}.json',result);print(json.dumps(result),flush=True)
    agent._estimate_value=original
    assert model_digest(agent.model)==before
    summary=dict(status='completed',condition=args.condition,task=args.task,rows=rows,
                 mean_return=sum(r['return'] for r in rows)/len(rows),
                 native_environment_steps=sum(r['steps'] for r in rows),
                 estimator_evaluations=sum(r['evaluations'] for r in rows),
                 additional_neural_evaluations_for_coupling=0,
                 parity_estimator_evaluations=2,model_hash=before,model_hash_unchanged=True,
                 elapsed_seconds=time.monotonic()-start,api_requests=0,training_updates=0,
                 gpu_work='Short inference only; CPU rank search included in elapsed time, no GPU utilization timing measurement.')
    save('summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--task',default='walker-walk');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--seeds',nargs='+',type=int,default=[17]);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['native','independent','targeted','shared','mean_heads'],required=True)
    main(p.parse_args())
