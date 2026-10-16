"""Change cross-candidate Gaussian dependence, preserving each particle law.

Given a unit vector q fixed by the past, eta~N(0,I) and independent z~N(0,1),
eta-q(q.T eta)+q*z is N(0,I). Correlating only z across candidate sequences
therefore preserves their full conditional N-particle laws. A fresh z is drawn
at every time step. This is a researcher-defined sampler intervention.
"""
import math

import torch


class GaussianCoupler:
    def __init__(self, wrapper, particles, mode, seed):
        self.wrapper=wrapper
        self.model=wrapper.model
        self.original=self.model.sample_1d
        self.particles=particles
        self.mode=mode
        self.rng=torch.Generator(device=self.model.device).manual_seed(seed)
        self.calls=0
        self.rows=0
        self.phases=None

    def install(self):
        self.model.sample_1d=self.sample

    def restore(self):
        self.model.sample_1d=self.original

    def sample(self, inputs, state, deterministic=False, rng=None):
        if deterministic or self.model.deterministic:
            return self.original(inputs,state,deterministic=deterministic,rng=rng)
        means,logvars=self.model.forward(inputs,rng=rng,propagation_indices=state['propagation_indices'])
        std=logvars.exp().sqrt()
        # Consume exactly the native Gaussian RNG operation; use an independent
        # generator for coupling so CEM's action proposal RNG is not modified.
        original_draw=torch.normal(means,std,generator=rng)
        eta=((original_draw-means)/std).reshape(-1,self.particles,means.shape[-1])
        current=inputs
        if self.wrapper.input_normalizer:
            current=self.wrapper.input_normalizer.denormalize(inputs)
        nxt=current[:,:4]+means
        # Smooth distance to the native failure boundaries is used ONLY by the
        # sampler. Native binary rewards, termination and CEM remain untouched.
        gradient=torch.zeros_like(means)
        gradient[:,0]=-2*nxt[:,0]/2.4**2
        gradient[:,2]=-2*nxt[:,2]/(12*2*math.pi/360)**2
        direction=(gradient*std).reshape_as(eta)
        norm=torch.linalg.vector_norm(direction,dim=(1,2),keepdim=True)
        fallback=torch.zeros_like(direction);fallback[:,0,2]=1
        q=torch.where(norm>1e-12,direction/norm.clamp_min(1e-12),fallback)
        score=-(nxt[:,0]/2.4).square()-(nxt[:,2]/(12*2*math.pi/360)).square()
        score=score.reshape(-1,self.particles).mean(1)
        candidates=len(score)
        if self.mode=='shared':
            phase=torch.zeros(candidates,device=means.device,dtype=torch.float64)
        elif self.mode=='antithetic':
            phase=(torch.arange(candidates,device=means.device)%2).double()*.5
        elif self.mode=='random':
            phase=torch.rand(candidates,generator=self.rng,device=means.device,dtype=torch.float64)
        elif self.mode=='targeted':
            # Favor the half with lower predicted safety margins on most draws;
            # the cyclic wraparound compensates exactly in their marginal laws.
            phase=(score<score.median()).double()*.1
        else:
            raise ValueError(self.mode)
        u=torch.rand((),generator=self.rng,device=means.device,dtype=torch.float64)
        uniforms=torch.remainder(u+phase,1.)
        z=torch.special.ndtri(uniforms.clamp(1e-15,1-1e-15)).to(eta.dtype)
        projection=(eta*q).sum((1,2),keepdim=True)
        coupled=eta-projection*q+z[:,None,None]*q
        self.calls+=1;self.rows+=means.shape[0]
        return means+std*coupled.reshape_as(means),state
