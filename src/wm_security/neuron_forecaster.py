"""Generic conductance-family fit to observed voltage, for shared acquisition comparisons.

No world registry, world name, discriminator, or test response enters fitting.
The public base dynamics and generic Chan parameterization come from NeuronBench.
Voltage synchronization is an optimization aid; forecasts always run freely.
This is a conventional system-identification baseline, not a proposed new method.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.stats import qmc
from neuronbench.worlds import build_I


@njit(cache=True)
def _rates(v):
    am=.1*(v+40)/(1-np.exp(-(v+40)/10)+1e-9)
    bm=4*np.exp(-(v+65)/18)
    ah=.07*np.exp(-(v+65)/20)
    bh=1/(1+np.exp(-(v+35)/10))
    an=.01*(v+55)/(1-np.exp(-(v+55)/10)+1e-9)
    bn=.125*np.exp(-(v+65)/80)
    return am,bm,ah,bh,an,bn


@njit(cache=True)
def _clip(x,lo,hi):
    return lo if x<lo else hi if x>hi else x


@njit(cache=True)
def integrate(I,p,channel,slow,sign,power,inactivation,obs_idx,obs_v,synchronize=False):
    """Native Euler order/step, optionally resetting voltage at observed samples."""
    dt=.01;v=-65.
    am,bm,ah,bh,an,bn=_rates(v)
    m=am/(am+bm);h=ah/(ah+bh);n=an/(an+bn);s=1.
    g,E,mvh,mk,mtau,hvh,hk,htau=p
    mk*=sign
    x=1/(1+np.exp(-(v-mvh)/mk));z=1/(1+np.exp((v-hvh)/hk)) if inactivation else 1.
    voltage=np.empty(len(I));j=0
    for i in range(len(I)):
        am,bm,ah,bh,an,bn=_rates(v)
        m=_clip(m+dt*(am*(1-m)-bm*m),0.,1.)
        h=_clip(h+dt*(ah*(1-h)-bh*h),0.,1.)
        n=_clip(n+dt*(an*(1-n)-bn*n),0.,1.)
        if slow:
            sinf=1/(1+np.exp((v+45)/4))
            tau=380. if v>-55. else 1500.
            s=_clip(s+dt*(sinf-s)/tau,0.,1.)
        current=120*m**3*h*(s if slow else 1.)*(v-50.)+36*n**4*(v+77.)+.3*(v+54.4)
        if channel:
            x=_clip(x+dt*(1/(1+np.exp(-(v-mvh)/mk))-x)/mtau,0.,1.)
            if inactivation:
                z=_clip(z+dt*(1/(1+np.exp((v-hvh)/hk))-z)/htau,0.,1.)
            current+=g*x**power*z*(v-E)
        v=_clip(v+dt*(I[i]-current),-95.,80.)
        voltage[i]=v
        if synchronize and j<len(obs_idx) and i==obs_idx[j]:
            v=obs_v[j];j+=1
    return voltage


LOW=np.array([np.log(.005),-100.,-110.,np.log(1.),np.log(.3),-115.,np.log(1.),np.log(1.)])
HIGH=np.array([np.log(30.),150.,0.,np.log(30.),np.log(500.),-25.,np.log(30.),np.log(3000.)])
EMPTY_I=np.empty(0,dtype=np.int64)
EMPTY_V=np.empty(0,dtype=float)


def decode(theta):
    p=np.asarray(theta).copy();p[[0,3,4,6,7]]=np.exp(p[[0,3,4,6,7]])
    return p


def model(p=None,channel=False,slow=False,sign=1,power=1,inactivation=False):
    if p is None:p=decode((LOW+HIGH)/2)
    return dict(parameters=np.asarray(p).tolist(),channel=bool(channel),slow=bool(slow),
                sign=int(sign),power=int(power),inactivation=bool(inactivation))


def trajectory(spec,segments):
    I,start=build_I(segments)
    V=integrate(I,np.array(spec['parameters']),spec['channel'],spec['slow'],spec['sign'],
                spec['power'],spec['inactivation'],EMPTY_I,EMPTY_V,False)
    previous=np.r_[-65.,V[:-1]]
    count=int(np.sum((previous[start:]<0)&(V[start:]>=0)))
    return count,V,start


def prepare(observations):
    return [(build_I(o['segments'])[0],np.array(o['obs_idx'],dtype=np.int64),np.array(o['voltage']))
            for o in observations]


def residual(spec,data,synchronize):
    values=[]
    for I,idx,V in data:
        pred=integrate(I,np.array(spec['parameters']),spec['channel'],spec['slow'],spec['sign'],
                       spec['power'],spec['inactivation'],idx,V,synchronize)
        values.append((pred[idx]-V)/math.sqrt(len(V)))
    return np.concatenate(values)/math.sqrt(len(data))


def fit(observations,seed=17,starts_per_structure=16,refine_per_structure=2,max_nfev=120,callback=None):
    data=prepare(observations);candidates=[];counts=dict(synchronized_simulations=0,free_simulations=0)
    def score(m,sync):
        counts['synchronized_simulations' if sync else 'free_simulations']+=len(data)
        return residual(m,data,sync)
    for slow in (False,True):
        m=model(slow=slow);r=score(m,False)
        candidates.append(dict(model=m,training_rmse=float(np.linalg.norm(r)),kind='public_base_family'))
    if min(c['training_rmse'] for c in candidates)<1e-7:
        return dict(selected=min(candidates,key=lambda c:c['training_rmse']),candidates=candidates,
                    counts=counts,early_exit='Public base-family model fits all observed voltage samples.')
    sobol=qmc.Sobol(d=8,scramble=True,seed=seed)
    points=LOW+(HIGH-LOW)*sobol.random_base2(int(np.ceil(np.log2(starts_per_structure))))[:starts_per_structure]
    for sign in (-1,1):
        for power in (1,2):
            for inactivation in (False,True):
                def construct(theta):
                    return model(decode(theta),channel=True,sign=sign,power=power,inactivation=inactivation)
                # Unused inactivation parameters stay fixed in the no-inactivation structure.
                indices=np.arange(8 if inactivation else 5)
                def objective(active):
                    theta=(LOW+HIGH)/2;theta[indices]=active
                    return score(construct(theta),True)
                screened=[]
                for p in points:
                    r=objective(p[indices]);screened.append((float(r@r),p[indices].copy()))
                for initial_rank,(_,initial) in enumerate(sorted(screened,key=lambda x:x[0])[:refine_per_structure]):
                    solved=least_squares(objective,initial,bounds=(LOW[indices],HIGH[indices]),
                                         x_scale='jac',max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                    theta=(LOW+HIGH)/2;theta[indices]=solved.x;m=construct(theta)
                    free_rmse=float(np.linalg.norm(score(m,False)))
                    c=dict(model=m,training_rmse=free_rmse,synchronized_rmse=float(np.linalg.norm(solved.fun)),
                           nfev=int(solved.nfev),converged=bool(solved.success),kind='generic_fitted_extra_current',
                           initial_rank=initial_rank)
                    candidates.append(c)
                    if callback:callback(c,counts)
    return dict(selected=min(candidates,key=lambda c:c['training_rmse']),candidates=candidates,counts=counts,
                early_exit=None)
