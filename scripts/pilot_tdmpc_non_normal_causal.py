"""Causal follow-up for the dose-300 non-normal cascade screen.

The source data and adaptation scope are held fixed.  The intervention is to
remove native MPPI at deployment and execute the learned actor (``prior``).
If the cascade effect is a planner-rollout mechanism, the actor should recover
while the source one-step error and weights remain identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_adaptation import adapt


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def episode_with_horizon(agent, task, seed, steps, horizon):
    """Run native MPPI after truncating its planning horizon."""
    cfg = agent.cfg
    old_horizon = cfg.horizon
    old_prev = agent._prev_mean
    cfg.horizon = horizon
    agent._prev_mean = torch.nn.Buffer(torch.zeros(horizon, cfg.action_dim, device=agent.device))
    try:
        result, data = episode(agent, task, seed, policy="mpc", steps=steps)
    finally:
        cfg.horizon = old_horizon
        agent._prev_mean = old_prev
    return result, data


def episode_with_std_floor(agent, task, seed, steps, floor):
    """Run native MPPI with a deployment-time entropy floor."""
    old_floor = agent.cfg.min_std
    agent.cfg.min_std = floor
    try:
        result, data = episode(agent, task, seed, policy="mpc", steps=steps)
    finally:
        agent.cfg.min_std = old_floor
    return result, data


def main(args):
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Replay the exact saved legal source trajectory and 300 dynamics-only native updates from the non-normal gate.",
        "causal_intervention": "Replace native MPPI deployment with the unchanged learned actor (prior) after adaptation.",
        "gate": "Cascade MPC harm must disappear or materially shrink under prior while random-control and clean actor returns remain stable.",
        "api_requests": 0,
    })
    rows = []
    for label, source_path, victim_task, eval_seed in args.arm:
        agent, provenance = load_agent(args.checkpoint)
        source = torch.load(ROOT / source_path, weights_only=False)
        history = adapt(agent, source, args.updates, seed=31, scope="dynamics")
        for policy in ["mpc", "prior"]:
            result, _ = episode(agent, victim_task, eval_seed, policy=policy, steps=args.eval_steps)
            row = {"arm": label, "policy": policy, **result,
                   "source_steps": len(source["action"]), "updates": args.updates,
                   "final_total_loss": history[-1].get("weighted_consistency_loss", history[-1].get("total_loss"))}
            rows.append(row)
            print(json.dumps(row), flush=True)
        result, _ = episode_with_horizon(agent, victim_task, eval_seed, args.eval_steps, args.causal_horizon)
        row = {"arm": label, **result, "policy": f"mpc_h{args.causal_horizon}",
               "source_steps": len(source["action"]), "updates": args.updates,
               "final_total_loss": history[-1].get("weighted_consistency_loss", history[-1].get("total_loss"))}
        rows.append(row)
        print(json.dumps(row), flush=True)
        result, _ = episode_with_std_floor(agent, victim_task, eval_seed, args.eval_steps, args.std_floor)
        row = {"arm": label, **result, "policy": f"mpc_floor{args.std_floor}",
               "source_steps": len(source["action"]), "updates": args.updates,
               "final_total_loss": history[-1].get("weighted_consistency_loss", history[-1].get("total_loss"))}
        rows.append(row)
        print(json.dumps(row), flush=True)
        save(out / f"{label}_adaptation.json", {"source": source_path, "history": history, "provenance": provenance})
        save(out / "results.json", rows)
    by = {(r["arm"], r["policy"]): r for r in rows}
    comparisons = []
    for label, _, _, _ in args.arm:
        mpc = by[(label, "mpc")]["return"]
        prior = by[(label, "prior")]["return"]
        comparisons.append({"arm": label, "mpc_return": mpc, "prior_return": prior,
                            "prior_minus_mpc": prior - mpc})
    save(out / "status.json", {
        "status": "completed", "results": rows, "comparisons": comparisons,
        "decision": "Planner-specific causal support is stronger when actor returns recover cascade MPC losses across both arms; this is not a defense or broad claim by itself.",
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--causal-horizon", type=int, default=1)
    p.add_argument("--std-floor", type=float, default=0.25)
    p.add_argument("--output", required=True)
    p.add_argument("--arm", nargs=4, action="append", metavar=("LABEL", "SOURCE", "TASK", "SEED"), required=True)
    a = p.parse_args()
    a.arm = [(x[0], x[1], x[2], int(x[3])) for x in a.arm]
    main(a)
