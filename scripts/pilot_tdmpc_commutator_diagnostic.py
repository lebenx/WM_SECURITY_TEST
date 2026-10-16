"""Measure action-order commutators after legal source adaptation.

This is a mechanism diagnostic only.  It does not infer a causal Lie-bracket
attack or task harm from a latent two-step residual.
"""
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error


def commutator(agent, data, task_name, count=64):
    task = agent.cfg.tasks.index(task_name)
    obs = data['obs'][:count].to(agent.device)
    z = agent.model.encode(obs, task)
    dim = agent.cfg.action_dims[task]
    g = torch.Generator(device=agent.device).manual_seed(811 + task)
    u = torch.rand((len(z), agent.cfg.action_dim), device=agent.device, generator=g) * 2 - 1
    v = torch.rand((len(z), agent.cfg.action_dim), device=agent.device, generator=g) * 2 - 1
    u[:, dim:] = 0; v[:, dim:] = 0
    uv = agent.model.next(agent.model.next(z, u, task), v, task)
    vu = agent.model.next(agent.model.next(z, v, task), u, task)
    return float(torch.linalg.vector_norm(uv - vu, dim=-1).mean().detach().cpu())


def main():
    out = Path('runs/tdmpc_commutator_diagnostic_v0'); out.mkdir(exist_ok=False)
    started = time.monotonic()
    base, provenance = load_agent('models/tdmpc2/mt30-48M.pt')
    source_cascade = collect(base, 'cheetah-run', 23, 300, 'cascade')
    source_random = collect(base, 'cheetah-run', 23, 300, 'random')
    tasks = ['walker-run', 'cup-catch', 'walker-walk']
    validation = {t: collect(base, t, 33, 300, 'random') for t in tasks}
    rows = []
    for condition, source in [('clean', None), ('random', source_random), ('cascade', source_cascade)]:
        agent, _ = load_agent('models/tdmpc2/mt30-48M.pt')
        if source is not None: adapt(agent, source, 300, 31, scope='dynamics')
        for task in tasks:
            rows.append(dict(condition=condition, task=task,
                             commutator=commutator(agent, validation[task], task),
                             one_step_error=one_step_error(agent, validation[task], task),
                             planner_horizon=agent.cfg.horizon))
    result = dict(status='completed', source_seed=23, updates=300, rows=rows,
                  interpretation='Commutator residual is a structural diagnostic; no non-commutative causal claim or task harm is inferred.',
                  elapsed_seconds=time.monotonic()-started, api_requests=0, gpu_work=True)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, default=str)+'\n')
    (out/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
