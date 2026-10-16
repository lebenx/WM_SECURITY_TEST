"""Short native gate for an action-Jacobian condition-number defense."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wm_security.tdmpc_native import episode, load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import frozen_digest
from screen_tdmpc_non_normal import collect, one_step_error


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def replay_actions(agent, actions, task_name, seed):
    set_seed(seed); idx = agent.cfg.tasks.index(task_name); env = make_one(task_name, seed)
    obs, rewards = [pad_obs(env.reset(), agent.cfg)], []
    try:
        for action in actions:
            nxt, reward, done, _ = env.step(action[:agent.cfg.action_dims[idx]])
            obs.append(pad_obs(nxt, agent.cfg)); rewards.append(float(reward))
            if done: break
    finally: env.unwrapped.close()
    if len(rewards) != len(actions): raise RuntimeError("replay ended early")
    return {"obs": torch.stack(obs), "action": actions.clone(), "reward": torch.tensor(rewards), "task": idx, "seed": seed}


def jacobian_ratio(model, z, action, task, create_graph):
    def f(a):
        return model.next(z.unsqueeze(0), a.unsqueeze(0), task)[0]
    jac = torch.autograd.functional.jacobian(f, action, create_graph=create_graph)
    sv = torch.linalg.svdvals(jac)
    return sv.max() / (sv.min() + 1e-6), float(sv.max().detach()), float(sv.min().detach())


def adapt_spectral(agent, data, updates, seed, beta, kappa):
    set_seed(seed); cfg, model = agent.cfg, agent.model
    for p in model.parameters(): p.requires_grad_(False)
    for p in model._dynamics.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.Adam(model._dynamics.parameters(), lr=cfg.lr)
    observations, actions, task = data["obs"].to(agent.device), data["action"].to(agent.device), data["task"]
    with torch.no_grad(): encoded = model.encode(observations, task)
    history = []
    for update in range(updates):
        starts = torch.randint(len(actions) - cfg.horizon + 1, (32,), device=agent.device)
        z = encoded[starts]; consistency = torch.zeros((), device=agent.device)
        for step in range(cfg.horizon):
            z = model.next(z, actions[starts + step], task)
            consistency = consistency + cfg.rho ** step * F.mse_loss(z, encoded[starts + step + 1])
        consistency = consistency / cfg.horizon * cfg.consistency_coef
        # One exact action Jacobian per update keeps this screen bounded while
        # preserving a differentiable spectral penalty.
        idx = int(starts[0].detach().cpu())
        ratio, largest, smallest = jacobian_ratio(
            model, encoded[idx].detach(), actions[idx].detach().clone().requires_grad_(True), task, True)
        penalty = beta * F.relu(ratio / kappa - 1.0)
        loss = consistency + penalty
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model._dynamics.parameters(), cfg.grad_clip_norm); optimizer.step()
        history.append({"update": update + 1, "consistency": float(consistency.detach()), "ratio": float(ratio.detach()),
                        "largest_sv": largest, "smallest_sv": smallest, "penalty": float(penalty.detach()), "grad_norm": float(grad)})
    return history


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    save(out / "plan.json", {**vars(args), "operation":"Native TD-MPC2 dynamics-only update plus action-Jacobian condition-number penalty; no replay/planner edits.", "api_requests":0})
    _, provenance = load_agent(args.checkpoint); save(out / "provenance.json", provenance)
    base = torch.load(args.cascade_source, weights_only=False)
    generator = torch.Generator().manual_seed(args.permutation_seed)
    perms = {"reverse_cascade": torch.arange(args.source_steps-1,-1,-1), "shuffled_cascade": torch.randperm(args.source_steps,generator=generator)}
    rows=[]
    for condition in ["clean","random","cascade","reverse_cascade","shuffled_cascade"]:
        agent,_=load_agent(args.checkpoint)
        if condition=="clean": source=None
        elif condition=="random": source=collect(agent,args.source_task,args.source_seed,args.source_steps,"random")
        elif condition=="cascade": source=base
        else: source=replay_actions(agent,base["action"][perms[condition]],args.source_task,args.source_seed)
        if source is not None:
            torch.save(source,out/f"{condition}_source.pt"); before=frozen_digest(agent.model)
            hist=adapt_spectral(agent,source,args.updates,args.learning_seed,args.beta,args.kappa)
            if before!=frozen_digest(agent.model): raise AssertionError("non-dynamics parameters changed")
            save(out/f"{condition}_adaptation.json",{"history":hist})
        validation=collect(agent,args.target_task,args.validation_seed,args.eval_steps,"random"); error=one_step_error(agent,validation,args.target_task)
        for seed in args.eval_seeds:
            result,_=episode(agent,args.target_task,seed,policy="mpc",steps=args.eval_steps)
            row={**result,"condition":condition,"one_step_error":error,"updates":0 if source is None else args.updates,"beta":args.beta,"kappa":args.kappa}
            rows.append(row); save(out/"results.json",rows); print(json.dumps(row),flush=True)
    by={(r["condition"],r["seed"]):r for r in rows}; gates=[]
    for seed in args.eval_seeds:
        rnd=by[("random",seed)]["return"]; cas=by[("cascade",seed)]["return"]
        gates.append({"seed":seed,"random_return":rnd,"cascade_return":cas,"reverse_return":by[("reverse_cascade",seed)]["return"],"shuffled_return":by[("shuffled_cascade",seed)]["return"],"cascade_loss":1-cas/max(rnd,1e-8),"one_step_error_ratio":by[("cascade",seed)]["one_step_error"]/max(by[("random",seed)]["one_step_error"],1e-12),"clean_return":by[("clean",seed)]["return"],"clean_vs_random_ratio":by[("clean",seed)]["return"]/max(rnd,1e-8)})
    save(out/"status.json",{"status":"completed","results":rows,"gates":gates,"elapsed_seconds":time.monotonic()-started,"decision":"Short Jacobian defense gate only; promote only after clean retention and two-seed recovery."})
    print(json.dumps({"status":"completed","gates":gates,"elapsed_seconds":time.monotonic()-started}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="models/tdmpc2/mt30-48M.pt"); p.add_argument("--cascade-source",default="runs/tdmpc_cascade_heldout_matched_v1/cascade_source.pt"); p.add_argument("--source-task",default="cheetah-run"); p.add_argument("--target-task",default="walker-run"); p.add_argument("--source-seed",type=int,default=23); p.add_argument("--validation-seed",type=int,default=33); p.add_argument("--eval-seeds",nargs="+",type=int,default=[31,32]); p.add_argument("--source-steps",type=int,default=300); p.add_argument("--eval-steps",type=int,default=200); p.add_argument("--updates",type=int,default=20); p.add_argument("--learning-seed",type=int,default=31); p.add_argument("--permutation-seed",type=int,default=933); p.add_argument("--beta",type=float,default=0.01); p.add_argument("--kappa",type=float,default=5.0); p.add_argument("--output",required=True); main(p.parse_args())
