"""Coordinate rank search using the unchanged competitors' elite threshold.

This avoids reranking every competitor for every trial value. Exact top-k is
still used to verify an accepted update and when a trial hits a score tie.
The resulting phases retain the same uniform-permutation sampling invariant.
"""
import torch


def optimize_rank_phases_fast(bank, elite_num, sweeps=2, max_coordinates=None):
    k,b=bank.shape
    ordered=bank.sort(0).values
    mean=bank.mean(0)
    rows=torch.arange(k)
    phases=torch.zeros(b,dtype=torch.long)
    current=ordered.clone()
    def exact(values):
        elites=values.topk(elite_num,dim=-1).indices
        return (.5*mean[elites[...,0]]+.5*mean[elites].mean(-1)).mean(-1)
    before=best=float(exact(current));evaluations=1;ties=0
    columns=torch.arange(b)
    if max_coordinates is not None and max_coordinates<b:
        columns=bank.max(0).values.topk(max_coordinates).indices
    columns=columns[mean[columns].argsort()].tolist()
    for _ in range(sweeps):
        changed=False
        for col in columns:
            others=current.clone();others[:,col]=-torch.inf
            other_values,other_indices=others.topk(elite_num,dim=-1)
            alternatives=ordered[(rows[None,:]+rows[:,None])%k,col]
            # If any values tie at a boundary, reproduce native topk's tie rule.
            ambiguous=(alternatives==other_values[:,-1]).any() or (alternatives==other_values[:,0]).any()
            if ambiguous:
                expanded=current.expand(k,k,b).clone();expanded[:,:,col]=alternatives
                values=exact(expanded);ties+=1
            else:
                first=torch.where(alternatives>other_values[:,0],mean[col],mean[other_indices[:,0]])
                elite_sum=mean[other_indices].sum(-1)
                inserted=(alternatives>other_values[:,-1]).float()
                new_sum=elite_sum+inserted*(mean[col]-mean[other_indices[:,-1]])
                values=(.5*first+.5*new_sum/elite_num).mean(-1)
            evaluations+=k
            choice=int(values.argmin())
            proposal=current.clone();proposal[:,col]=alternatives[choice]
            value=float(exact(proposal))
            if value<best-1e-7:
                phases[col]=choice;current=proposal;best=value;changed=True
        if not changed:break
    return phases,dict(before=before,after=best,candidates=evaluations,tie_fallbacks=ties)
