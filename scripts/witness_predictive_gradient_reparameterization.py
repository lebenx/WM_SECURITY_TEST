"""Reproduce a forward-law-preserving, discontinuous-reparameterization witness.

This is a mathematical test, not a native task or a novel gradient theorem.
Boundary terms are established prior work. Native security utility is unproven.
"""
import argparse
import json
from pathlib import Path

from scipy.stats import kstest
import torch


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    rng=torch.Generator().manual_seed(2026)
    def sample(a,n,attack):
        u=torch.rand(n,generator=rng)
        v=torch.remainder(u-10*a,1.) if attack else u
        assert bool(((v>0)&(v<1)).all())
        return a+.1*torch.special.ndtri(v)
    observations=[]
    for attack in [False,True]:
        a=torch.tensor(.137,requires_grad=True)
        y=sample(a,200000,attack);g=torch.autograd.grad(y.mean(),a)[0]
        observations.append(dict(condition='shifted_quantile' if attack else 'native_gaussian',
            sample_mean=float(y.mean().detach()),sample_std=float(y.std().detach()),
            ks_distance_to_native_gaussian=float(kstest((y.detach().numpy()-.137)/.1,'norm').statistic),
            autodiff_mean_gradient=float(g),true_expectation_gradient=1.))
    optimization=[]
    for attack in [False,True]:
        a=torch.tensor(0.,requires_grad=True);trace=[]
        for step in range(40):
            y=sample(a,64,attack);g=torch.autograd.grad(y.mean(),a)[0]
            with torch.no_grad():a.add_(.05*g).clamp_(-1,1)
            trace.append(dict(step=step,a=float(a.detach()),gradient=float(g)))
        optimization.append(dict(condition='shifted_quantile' if attack else 'native_gaussian',
            final_action=float(a.detach()),true_expected_reward=float(a.detach()),trace=trace))
    a=torch.tensor(.137);h=.01
    finite_difference=float((sample(a+h,500000,True).mean()-sample(a-h,500000,True).mean())/(2*h))
    result=dict(status='analytic_and_autodiff_witness_only',observations=observations,
        optimization=optimization,independent_sample_finite_difference_gradient=finite_difference,
        finite_difference_step=h,
        law='For each fixed action, modulo shifts preserve a fresh uniform law, so both samplers have law N(a,0.01). Fresh independent uniforms preserve the full conditional joint forward law.',
        explanation='True expected-reward derivative is +1; ordinary autodiff omits moving-boundary terms in the discontinuous quantile map. This known gradient-estimation phenomenon is not a novelty claim.',
        limits='No custom backward, no native environment, no learned-model attack, no meaningful novelty conclusion.',
        native_environment_steps=0,api_requests=0,gpu_work=False,training_updates=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(observations=observations,finite_difference_gradient=finite_difference)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(p.parse_args().output)
