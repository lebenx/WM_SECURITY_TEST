"""Causal action-order controls for the strongest cascade direction.

The source action vectors are kept fixed.  Each permutation is replayed in a
fresh simulator so observations/rewards remain truthful, then the resulting
legal trajectory is used for dynamics-only adaptation.  The held-out victim
uses unchanged native MPC and the learned actor as a planner-specific control.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import episode, load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt, frozen_digest
from screen_tdmpc_non_normal import collect, one_step_error


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def replay_actions(agent, actions, task_name, seed):
    set_seed(seed)
    idx = agent.cfg.tasks.index(task_name)
    env = make_one(task_name, seed)
    observations, rewards = [pad_obs(env.reset(), agent.cfg)], []
    try:
        for action in actions:
            nxt, reward, done, _ = env.step(action[:agent.cfg.action_dims[idx]])
            observations.append(pad_obs(nxt, agent.cfg))
            rewards.append(float(reward))
            if done:
                break
    finally:
        env.unwrapped.close()
    if len(rewards) != len(actions):
        raise RuntimeError(f"replay ended early: {len(rewards)} != {len(actions)}")
    return {"obs": torch.stack(observations), "action": actions.clone(),
            "reward": torch.tensor(rewards), "task": idx, "seed": seed}


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Replay the same legal source action vectors under temporal permutations; dynamics-only native TD-MPC2 adaptation; held-out native MPC and actor evaluation.",
        "conditions": ["random", "cascade", "reverse_cascade", "shuffled_cascade"],
        "gate": "A directional order mechanism requires cascade harm against random and shuffled controls, low one-step error, and materially smaller actor loss than MPC loss.",
        "api_requests": 0,
        "gpu_work": "bounded native CUDA only",
    })
    started = time.monotonic()
    base_agent, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    cascade = torch.load(args.cascade_source, weights_only=False)
    if len(cascade["action"]) != args.source_steps:
        raise ValueError("cascade source length differs from --source-steps")
    generator = torch.Generator().manual_seed(args.permutation_seed)
    perms = {
        "reverse_cascade": torch.arange(args.source_steps - 1, -1, -1),
        "shuffled_cascade": torch.randperm(args.source_steps, generator=generator),
    }
    rows = []
    for condition in ["clean", "random", "cascade", "reverse_cascade", "shuffled_cascade"]:
        agent, _ = load_agent(args.checkpoint)
        if condition == "clean":
            source = None
        elif condition == "random":
            source = collect(agent, args.source_task, args.source_seed, args.source_steps, "random")
        elif condition == "cascade":
            source = cascade
        else:
            source = replay_actions(agent, cascade["action"][perms[condition]], args.source_task, args.source_seed)
        if source is not None:
            torch.save(source, out / f"{condition}_source.pt")
            before = frozen_digest(agent.model)
            history = adapt(agent, source, args.updates, args.learning_seed, "dynamics")
            if before != frozen_digest(agent.model):
                raise AssertionError("non-dynamics parameters changed")
            save(out / f"{condition}_adaptation.json", {"history": history})
        validation = collect(agent, args.target_task, args.validation_seed, args.eval_steps, "random")
        error = one_step_error(agent, validation, args.target_task)
        for seed in args.eval_seeds:
            for policy in ("mpc", "prior"):
                result, _ = episode(agent, args.target_task, seed, policy=policy, steps=args.eval_steps)
                row = {**result, "condition": condition, "policy": policy,
                       "one_step_error": error, "planner_horizon": agent.cfg.horizon,
                       "source_steps": 0 if source is None else len(source["action"])}
                rows.append(row)
                save(out / "results.json", rows)
                print(json.dumps(row), flush=True)
    by = {(r["condition"], r["policy"], r["seed"]): r for r in rows}
    comparisons = []
    for seed in args.eval_seeds:
        for condition in ["cascade", "reverse_cascade", "shuffled_cascade", "random"]:
            comparisons.append({
                "seed": seed, "condition": condition,
                "mpc_return": by[(condition, "mpc", seed)]["return"],
                "actor_return": by[(condition, "prior", seed)]["return"],
                "mpc_minus_actor": by[(condition, "mpc", seed)]["return"] - by[(condition, "prior", seed)]["return"],
                "one_step_error": by[(condition, "mpc", seed)]["one_step_error"],
                "mpc_loss_vs_random": 1.0 - by[(condition, "mpc", seed)]["return"] / max(by[("random", "mpc", seed)]["return"], 1e-8),
            })
    save(out / "status.json", {
        "status": "completed", "results": rows, "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "decision": "Order-control diagnostic only. Promote only after an independent source seed, full matched shuffled control, and a useful defense.",
    })
    print(json.dumps({"status": "completed", "comparisons": comparisons,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_cascade_heldout_matched_v1/cascade_source.pt")
    p.add_argument("--source-task", default="cheetah-run")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--source-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--permutation-seed", type=int, default=933)
    p.add_argument("--output", required=True)
    main(p.parse_args())
