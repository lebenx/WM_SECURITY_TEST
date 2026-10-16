"""Native post-adaptation SimNorm temperature intervention pilot."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error


class TemperatureSimNorm(nn.Module):
    def __init__(self, dim, temperature):
        super().__init__()
        self.dim = dim
        self.temperature = float(temperature)

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        return F.softmax(x / self.temperature, dim=-1).view(*shp)


def patch_temperature(agent, temperature):
    if temperature == 1.0:
        return
    act = agent.model._dynamics[-1].act
    patch_temperature.original_dim = act.dim
    agent.model._dynamics[-1].act = TemperatureSimNorm(act.dim, temperature)


def main():
    out = Path('runs/tdmpc_simnorm_temperature_v1')
    out.mkdir(exist_ok=False)
    plan = dict(source_task='cheetah-run', source_seed=23, eval_task='walker-run',
                eval_seed=31, validation_seed=33, updates=300,
                temperatures=[1.0, 2.0], conditions=['clean', 'random', 'cascade'],
                intervention='After each condition adapts with native dynamics-only objective under SimNorm temperature 1, replace only dynamics final SimNorm with softmax(x/temperature) before native MPC evaluation.',
                gate='Exploratory causal screen: temperature 2 should recover cascade return while preserving clean/random return and low one-step error; no broad claim from this pilot.', api_requests=0)
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.monotonic()
    base, provenance = load_agent('models/tdmpc2/mt30-48M.pt')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    source = collect(base, 'cheetah-run', 23, 300, 'cascade')
    random_source = collect(base, 'cheetah-run', 23, 300, 'random')
    validation = collect(base, 'walker-run', 33, 300, 'random')
    torch.save(source, out/'cascade_source.pt')
    torch.save(random_source, out/'random_source.pt')
    rows = []
    for temp in plan['temperatures']:
        for condition, data in [('clean', None), ('random', random_source), ('cascade', source)]:
            agent, _ = load_agent('models/tdmpc2/mt30-48M.pt')
            if data is not None:
                adapt(agent, data, 300, 31, scope='dynamics')
            patch_temperature(agent, temp)
            err = one_step_error(agent, validation, 'walker-run')
            result, _ = episode(agent, 'walker-run', 31, steps=500)
            row = dict(result, condition=condition, temperature=temp,
                       one_step_error=err, planner_horizon=agent.cfg.horizon)
            rows.append(row)
            (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    status = dict(status='completed', results=rows, elapsed_seconds=time.monotonic()-started,
                  interpretation='Temperature intervention is only a causal diagnostic. Compare clean and random controls before attributing any selective recovery to latent non-normality.')
    (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
