"""Full native update order control with the original actor restored for MPC."""
from __future__ import annotations
import argparse, json, time
from copy import deepcopy
from pathlib import Path
import torch
from wm_security.tdmpc_native import episode, load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error

def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=float)+"\n")

def main(a):
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True); start=time.monotonic(); rows=[]
    sources=[("random",a.random_source),("cascade",a.cascade_source),("reverse_cascade",a.reverse_source),("shuffled_cascade",a.shuffled_source)]
    for condition,path in sources:
        agent,prov=load_agent(a.checkpoint)
        if condition=="random": save(out/"provenance.json",prov)
        source=torch.load(path,weights_only=False)
        actor_state={k:v.detach().cpu().clone() for k,v in agent.model._pi.state_dict().items()}
        history=adapt(agent,source,a.updates,a.learning_seed,"native_full")
        agent.model._pi.load_state_dict(actor_state)
        save(out/f"{condition}_adaptation.json",{"history":history,"actor_restored":True})
        validation=collect(agent,a.target_task,a.validation_seed,a.eval_steps,"random")
        err=one_step_error(agent,validation,a.target_task)
        for seed in a.eval_seeds:
            result,_=episode(agent,a.target_task,seed,policy="mpc",steps=a.eval_steps)
            row={**result,"condition":condition,"one_step_error":err,"actor_restored":True,"updates":a.updates}
            rows.append(row); save(out/"results.json",rows); print(json.dumps(row),flush=True)
    by={(r["condition"],r["seed"]):r for r in rows}; comps=[]
    for seed in a.eval_seeds:
        rnd=by[("random",seed)]
        for c in ["cascade","reverse_cascade","shuffled_cascade"]:
            x=by[(c,seed)]; comps.append({"seed":seed,"condition":c,"mpc_return":x["return"],"random_return":rnd["return"],"loss_vs_random":1-x["return"]/max(rnd["return"],1e-8),"error_ratio_vs_random":x["one_step_error"]/max(rnd["one_step_error"],1e-12)})
    save(out/"status.json",{"status":"completed","results":rows,"comparisons":comps,"elapsed_seconds":time.monotonic()-start,"decision":"Causal actor-restore strong baseline; retain only if order-specific MPC loss survives actor restoration with low error."})
    print(json.dumps({"status":"completed","comparisons":comps,"elapsed_seconds":time.monotonic()-start}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt'); p.add_argument('--random-source',default='runs/tdmpc_order_controls_v0/random_source.pt'); p.add_argument('--cascade-source',default='runs/tdmpc_order_controls_v0/cascade_source.pt'); p.add_argument('--reverse-source',default='runs/tdmpc_order_controls_v0/reverse_cascade_source.pt'); p.add_argument('--shuffled-source',default='runs/tdmpc_order_controls_v0/shuffled_cascade_source.pt'); p.add_argument('--target-task',default='walker-run'); p.add_argument('--validation-seed',type=int,default=33); p.add_argument('--eval-seeds',nargs='+',type=int,default=[31,32]); p.add_argument('--eval-steps',type=int,default=300); p.add_argument('--updates',type=int,default=80); p.add_argument('--learning-seed',type=int,default=31); p.add_argument('--output',required=True); main(p.parse_args())
