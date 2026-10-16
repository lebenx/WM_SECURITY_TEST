"""Bounded update-exposure test of already fixed genuine curricula.

No extra action search. Data, model, optimizer and test seed are held fixed;
only the number of native update calls differs. The300-update endpoint matches
one update per newly collected action, but batch32 and isolated recent-data
adaptation remain researcher-defined settings.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_action_search import snapshot, reset_agent, source_error
from pilot_tdmpc_adaptation import adapt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='runs/tdmpc_action_search_v0')
    p.add_argument('--selection',default='runs/tdmpc_return_search_v0/selection.json')
    p.add_argument('--updates',type=int,nargs='+',default=[120,300])
    p.add_argument('--seed',type=int,default=12)
    p.add_argument('--output',default='runs/tdmpc_update_dose_v0')
    args=p.parse_args();root=Path(args.source);out=Path(args.output)
    if any(n<1 or n>300 for n in args.updates) or len(args.updates)>2:
        raise ValueError('This pilot allows at most two update counts, each at most300.')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    source=json.loads((root/'plan.json').read_text())
    selection=json.loads(Path(args.selection).read_text())
    files={'benign':root/'benign_deployed_data.pt','random':root/'random_deployed_data.pt',
           'max_error':root/f'candidate_{selection["max_error"]:02d}.pt',
           'targeted':root/f'candidate_{selection["selected"]:02d}.pt'}
    plan=dict(vars(args),source_plan=source,selection=selection,files={k:str(v) for k,v in files.items()},
              scope='Fixed action curricula from completed search. Only update exposure changes. Fresh learner per arm/dose; original full update at batch32 and lr3e-4. One development target reset, no new search or population claim.',
              native_schedule_source='vendor/tdmpc2/tdmpc2/trainer/online_trainer.py:115-122',
              update_exposure_note='300 updates for300 new physical transitions is a1:1 count ratio. This is not the upstream replay-buffer protocol: only the recent source contribution is replayed here.')
    save('plan.json',plan);results=[];started=time.monotonic()
    try:
        agent,prov=load_agent(source['checkpoint']);save('provenance.json',prov)
        original=snapshot(agent)
        result,data=episode(agent,source['victim_task'],args.seed)
        result.update(condition='clean',updates=0);results.append(result)
        save('results.json',results);print(json.dumps(result),flush=True)
        for count in args.updates:
            for condition,path in files.items():
                reset_agent(agent,original)
                data=torch.load(path,weights_only=False)
                source_error(agent,data)
                start=time.monotonic()
                history=adapt(agent,data,count,31,'native_full')
                save(f'{condition}_{count}_learning.json',{'seconds':time.monotonic()-start,'history':history})
                result,episode_data=episode(agent,source['victim_task'],args.seed)
                result.update(condition=condition,updates=count);results.append(result)
                torch.save(episode_data,out/f'{condition}_{count}_evaluation.pt')
                if result['return'] < 800:
                    torch.save({'model':agent.model.state_dict()},out/f'{condition}_{count}_model.pt')
                save('results.json',results);print(json.dumps(result),flush=True)
        save('status.json',{'status':'completed','results':results,'wall_seconds':time.monotonic()-started,
                            'interpretation':'Exposure sensitivity on fixed genuine curricula. No new optimized attack, no held-out instance aggregate, no background-data mixing or ordinary-user repair yet.'})
    except Exception as exc:
        save('status.json',{'status':'error','error':str(exc),'traceback':traceback.format_exc(),'results':results});raise


if __name__=='__main__':main()
