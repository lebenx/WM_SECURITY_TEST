"""Fresh-bank rank coupling preserves each candidate's native forecast law.

For every candidate, generate K complete native prediction batches. Conditional
on this entire bank, any cyclic permutation of a fresh uniform row index still
selects each batch with probability 1/K. The permutation can depend on the bank.
Unconditionally the returned batch therefore has exactly the original law.
Only the joint law across candidate actions changes. No forecast is edited.
"""
import torch


def optimize_rank_phases(bank,elite_num,sweeps=2):
    k,b=bank.shape
    ordered=bank.sort(dim=0).values
    mean=bank.mean(0)
    rows=torch.arange(k)
    phases=torch.zeros(b,dtype=torch.long)
    current=ordered.clone()
    def objective(values):
        elites=values.topk(elite_num,dim=-1).indices
        quality=.5*mean[elites[...,0]]+.5*mean[elites].mean(-1)
        return quality.mean(-1)
    before=float(objective(current));best=before;evaluations=1
    # Lower-mean candidates are considered first; this order is bank-dependent
    # but chosen before drawing the returned row, so preservation still holds.
    for _ in range(sweeps):
        changed=False
        for col in mean.argsort().tolist():
            alternatives=current.expand(k,k,b).clone()
            alternatives[:,:,col]=ordered[(rows[None,:]+rows[:,None])%k,col]
            values=objective(alternatives);evaluations+=k
            choice=int(values.argmin());value=float(values[choice])
            if value<best-1e-7:
                phases[col]=choice;current=alternatives[choice].clone();best=value;changed=True
        if not changed:break
    return phases,dict(before=before,after=best,candidates=evaluations)


class RankBankSampler:
    def __init__(self,model_env,particles,bank_size,condition,seed,elite_num=10):
        self.env=model_env;self.particles=particles;self.bank_size=bank_size
        self.condition=condition;self.elite_num=elite_num
        self.rng=torch.Generator().manual_seed(seed)
        self.calls=0;self.predictive_transitions=0;self.phase_candidates=0
        self.surrogate_before=0.;self.surrogate_after=0.

    def evaluate(self,initial_state,actions):
        bank=torch.stack([self.env.evaluate_action_sequences(actions,initial_state,self.particles)
                          for _ in range(self.bank_size)])
        self.calls+=1
        self.predictive_transitions+=self.bank_size*len(actions)*actions.shape[1]*self.particles
        ordered=bank.sort(dim=0).values
        if self.condition=='independent':
            indices=torch.randint(self.bank_size,(len(actions),),generator=self.rng)
        else:
            if self.condition=='targeted':
                phases,info=optimize_rank_phases(bank,self.elite_num)
                self.phase_candidates+=info['candidates']
                self.surrogate_before+=info['before'];self.surrogate_after+=info['after']
            elif self.condition=='shared':
                phases=torch.zeros(len(actions),dtype=torch.long)
            else:raise ValueError(self.condition)
            r=torch.randint(self.bank_size,(),generator=self.rng)
            indices=(r+phases)%self.bank_size
        return ordered[indices,torch.arange(len(actions))]
