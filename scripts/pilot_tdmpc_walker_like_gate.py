"""Bounded held-out walker-like confirmation for the legal cascade screen.

This keeps the already fixed cheetah-run source trajectories and dynamics-only
300-update dose.  It adds no search or planner changes; each target is evaluated
with clean, random-source, and cascade-source learners from the same checkpoint.
"""
import argparse, json, time
from pathlib import Path
import torch
from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cascade-source', default='runs/tdmpc_non_normal_gate_cheetah_run_dose300_v0/cascade_source.pt')
    p.add_argument('--random-source', default='runs/tdmpc_non_normal_gate_cheetah_run_dose300_v0/random_source.pt')
    p.add_argument('--targets', nargs='+', default=['walker-stand', 'walker-walk'])
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--output', required=True)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=False, exist_ok=False)
    plan = {**vars(a), 'operation': 'Fixed legal cheetah-run cascade/random source trajectories; dynamics-only native TD-MPC2 consistency update; no search or planner modification.', 'api_requests': 0, 'gpu_work': True}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results=[]; started=time.monotonic()
    for target in a.targets:
        clean_agent,_=load_agent('models/tdmpc2/mt30-48M.pt')
        clean,_=episode(clean_agent,target,a.seed,policy='mpc',steps=a.steps)
        results.append({'target':target,'condition':'clean',**clean})
        for condition,source_path in [('random',a.random_source),('cascade',a.cascade_source)]:
            agent,_=load_agent('models/tdmpc2/mt30-48M.pt')
            source=torch.load(source_path,weights_only=False)
            history=adapt(agent,source,a.updates,seed=31,scope='dynamics')
            result,_=episode(agent,target,a.seed,policy='mpc',steps=a.steps)
            row={'target':target,'condition':condition,**result,
                 'source_path':source_path,'updates':a.updates,
                 'final_consistency_loss':history[-1]['weighted_consistency_loss']}
            results.append(row)
            (out/f'{condition}_{target}_adaptation.json').write_text(json.dumps({'history':history},indent=2)+'\n')
            print(json.dumps(row),flush=True)
    by={(r['target'],r['condition']):r for r in results}
    comparisons=[]
    for target in a.targets:
        comparisons.append({'target':target,'clean':by[(target,'clean')]['return'],
                            'random':by[(target,'random')]['return'],
                            'cascade':by[(target,'cascade')]['return'],
                            'cascade_vs_random_drop':by[(target,'random')]['return']-by[(target,'cascade')]['return']})
    status={'status':'completed','results':results,'comparisons':comparisons,
            'interpretation':'Held-out walker-like family screen. A positive result would still require two seeds, matched shuffled controls, causal intervention and a useful defense before promotion.',
            'elapsed_seconds':time.monotonic()-started,'api_requests':0,'gpu_work':True}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'status':status['status'],'comparisons':comparisons,'elapsed_seconds':status['elapsed_seconds']}),flush=True)


if __name__ == '__main__': main()
