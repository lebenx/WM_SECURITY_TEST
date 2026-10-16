"""Whole-rollout Gaussian coupling; a bounded surrogate-based construction.

The direction and phases are fixed before drawing an entire rollout's noise.
Projection replacement preserves the full H*N*D standard-normal marginal for
each candidate, conditional on its native fixed ensemble-member allocation.
"""
import math

import numpy as np
from scipy.special import ndtri
import torch
from torch.nn import functional as F


class TrajectoryCoupler:
    def __init__(self,wrapper,model_env,particles,mode,seed):
        self.wrapper=wrapper;self.model=wrapper.model;self.env=model_env
        self.particles=particles;self.mode=mode
        self.rng=torch.Generator().manual_seed(seed)
        self.original_sample=self.model.sample_1d;self.original_reset=self.env.reset
        self.actions=None;self.noise=None;self.step=0
        self.surrogate_forward_transitions=0;self.copula_candidates=0

    def install(self):
        self.model.sample_1d=self.sample;self.env.reset=self.reset

    def restore(self):
        self.model.sample_1d=self.original_sample;self.env.reset=self.original_reset

    def reset(self,initial_obs_batch,return_as_np=True):
        state=self.original_reset(initial_obs_batch,return_as_np=return_as_np)
        self.prepare(state)
        return state

    def choose_phase(self,mu,sigma):
        n=len(mu)
        if self.mode=='shared':return np.zeros(n)
        if self.mode=='spread':return np.arange(n)/n
        rank=np.argsort(np.argsort(-mu))
        candidates=[np.zeros(n),np.arange(n)/n,rank/n]
        for fraction in [.1,.25,.5,.75]:
            cutoff=max(1,int(fraction*n))
            for width in [.1,.3,1.]:
                candidates.append(np.where(rank<cutoff,0.,width*(rank-cutoff)/(n-cutoff)))
        for shift in [.05,.1,.25,.5,.9]:candidates.append(np.where(rank<n//2,0.,shift))
        # Fixed integration grid only selects a coupling; actual sampling below
        # uses a fresh continuous uniform, never this deterministic grid.
        u=(np.arange(64)+.5)/64
        phase=np.asarray(candidates)
        z=ndtri(np.mod(phase[:,:,None]+u[None,None,:],1.))
        estimates=mu[None,:,None]+sigma[None,:,None]*z
        winner=estimates.argmax(axis=1)
        objective=mu[winner].mean(axis=1)
        self.copula_candidates+=len(candidates)
        return phase[int(objective.argmin())]

    def prepare(self,state):
        batch,horizon,_=self.actions.shape;n=self.particles;dim=4
        # Differentiation is used by the sampler's construction, not by native
        # CEM, its rewards, its termination decisions or the deployed model.
        with torch.enable_grad():
            eps=torch.zeros(horizon,batch*n,dim,requires_grad=True)
            obs=state['obs'].detach();log_survival=torch.zeros(batch*n)
            score=torch.zeros(batch*n)
            for t in range(horizon):
                act=self.actions[:,t].repeat_interleave(n,0)
                inputs=self.wrapper._get_model_input(obs,act)
                mean,logvar=self.model.forward(inputs,propagation_indices=state['propagation_indices'])
                obs=obs+mean+logvar.exp().sqrt()*eps[t]
                # Differentiable approximation to upstream CartPole survival.
                log_survival=log_survival+F.logsigmoid(4*(1-(obs[:,0]/2.4).square()))
                log_survival=log_survival+F.logsigmoid(4*(1-(obs[:,2]/(12*2*math.pi/360)).square()))
                score=score+log_survival.exp()
            mu=score.reshape(batch,n).mean(1)
            gradient=torch.autograd.grad(mu.sum(),eps)[0]
        self.surrogate_forward_transitions+=batch*n*horizon
        direction=gradient.reshape(horizon,batch,n,dim).permute(1,0,2,3).reshape(batch,-1).detach()
        norm=torch.linalg.vector_norm(direction,dim=1,keepdim=True)
        fallback=torch.zeros_like(direction);fallback[:,0]=1.
        q=torch.where(norm>1e-12,direction/norm.clamp_min(1e-12),fallback)
        phase=self.choose_phase(mu.detach().numpy(),norm[:,0].numpy())
        eta=torch.randn(q.shape,generator=self.rng)
        if self.mode=='independent':
            z=torch.randn(batch,generator=self.rng)
        else:
            u=torch.rand((),generator=self.rng,dtype=torch.float64)
            z=torch.special.ndtri(torch.remainder(u+torch.as_tensor(phase),1.).clamp(1e-15,1-1e-15)).float()
        noise=eta-q*(eta*q).sum(1,keepdim=True)+q*z[:,None]
        self.noise=noise.reshape(batch,horizon,n,dim).permute(1,0,2,3).reshape(horizon,batch*n,dim)
        self.step=0

    def sample(self,inputs,state,deterministic=False,rng=None):
        if deterministic or self.model.deterministic:
            return self.original_sample(inputs,state,deterministic=deterministic,rng=rng)
        mean,logvar=self.model.forward(inputs,rng=rng,propagation_indices=state['propagation_indices'])
        std=logvar.exp().sqrt()
        # Advance the native model RNG identically, preserving its separation
        # from CEM and the native member-allocation RNG; these draws are replaced.
        torch.normal(mean,std,generator=rng)
        noise=self.noise[self.step];self.step+=1
        return mean+std*noise,state
