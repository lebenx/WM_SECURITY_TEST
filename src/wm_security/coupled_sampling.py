"""Finite-budget planning with unchanged per-action particle laws.

For each action, inverse-CDF sampling of multinomial particle counts followed by
a uniform permutation has exactly the law of N independent categorical draws.
Replacing each action's uniform by (U + phase[action]) mod 1 changes only the
dependence ACROSS actions. It preserves even the full N-particle marginal, not
merely the next-state mean or variance. This module uses known finite laws to
test that boundary; it does not train or compromise a neural world model.
"""
from itertools import product
import math

import numpy as np


def particle_law(outcomes, probabilities, particles):
    """Three native outcomes, including any duplicate next states."""
    outcomes=np.asarray(outcomes,dtype=float)
    p=np.asarray(probabilities,dtype=float)
    assert len(p)==3 and np.isclose(p.sum(),1)
    rows=[]
    for i in range(particles+1):
        for j in range(particles-i+1):
            counts=np.array([i,j,particles-i-j])
            probability=math.comb(particles,i)*math.comb(particles-i,j)*np.prod(p**counts)
            if probability>0:
                rows.append((round(float(counts@outcomes/particles),12),float(probability),counts))
    rows.sort(key=lambda x:x[0])
    values=np.array([r[0] for r in rows]);weights=np.array([r[1] for r in rows])
    weights/=weights.sum();cdf=np.cumsum(weights);cdf[-1]=1.
    return dict(values=values,weights=weights,cdf=cdf,counts=np.array([r[2] for r in rows]),
                probabilities=p,outcomes=outcomes,particles=particles)


def select(laws, uniforms):
    values=np.array([d['values'][np.searchsorted(d['cdf'],u,side='right')]
                     for d,u in zip(laws,uniforms)])
    return int(np.argmax(values))


def draw_particle_indices(law,u,rng):
    row=np.searchsorted(law['cdf'],u,side='right')
    result=np.repeat(np.arange(3),law['counts'][row])
    rng.shuffle(result)
    return result


def coupled_policy(laws, phases, check_marginals=False):
    """Integrate exactly over a single uniform; no simulation or discretization."""
    cuts=np.unique(np.r_[0.,1.,*[np.mod(np.r_[0.,d['cdf']]-z,1.) for d,z in zip(laws,phases)]])
    width=np.diff(cuts);mid=(cuts[:-1]+cuts[1:])/2
    values=[];indices=[]
    for d,z in zip(laws,phases):
        idx=np.searchsorted(d['cdf'],np.mod(mid+z,1.),side='right')
        indices.append(idx);values.append(d['values'][idx])
    decisions=np.argmax(np.array(values),axis=0)
    policy=np.bincount(decisions,weights=width,minlength=len(laws))
    max_error=0.
    if check_marginals:
        for d,idx in zip(laws,indices):
            recovered=np.bincount(idx,weights=width,minlength=len(d['weights']))
            max_error=max(max_error,float(np.max(np.abs(recovered-d['weights']))))
    return policy,max_error


def independent_policy(laws):
    """Exact probability of each argmax under independent action samples."""
    policy=[]
    for a,d in enumerate(laws):
        win=d['weights'].copy()
        for b,e in enumerate(laws):
            if a==b:continue
            # Earlier action indices win exact ties.
            ids=np.searchsorted(e['values'],d['values'],side='left' if b<a else 'right')
            win*=np.r_[0.,e['cdf']][ids]
        policy.append(float(win.sum()))
    result=np.array(policy)
    assert abs(result.sum()-1)<1e-9
    return result


def optimize_phases(laws, true_q, seed=17):
    """Bounded coordinate search; minimize selected-action value under fixed V*."""
    rng=np.random.default_rng(seed)
    candidates=[np.zeros(len(laws)),*rng.random((24,len(laws)))]
    evaluations=0
    def objective(z):
        nonlocal evaluations
        evaluations+=1
        return float(coupled_policy(laws,z)[0]@true_q)
    best=min(candidates,key=objective);best_value=objective(best)
    for _ in range(2):
        changed=False
        for a in range(len(laws)):
            for x in np.arange(16)/16:
                z=best.copy();z[a]=x
                value=objective(z)
                if value<best_value-1e-14:
                    best=z;best_value=value;changed=True
        if not changed:break
    return best,dict(objective=best_value,evaluations=evaluations)
