"""Causal short gate for shared terminal-value provenance contamination.

Only the shared online Q head is updated from authentic source-task
transitions. Encoder, latent dynamics, reward model, task embedding, target
policy, and actor are frozen.  A positive result would isolate a planner-only
cross-task mechanism with exactly unchanged one-step dynamics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import torch

from wm_security.tdmpc_native import episode, load_agent


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def adapt_q_only(agent, data, updates: int, seed: int):
    """Run upstream value loss while leaving every non-Q module untouched."""
    from pilot_tdmpc_adaptation import set_seed

    set_seed(seed)
    cfg, model = agent.cfg, agent.model
    cfg.batch_size = 32
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model._Qs.parameters():
        p.requires_grad_(True)
    # The upstream _update also updates the actor.  Replacing that substep with
    # a no-op keeps the target policy and all actor parameters byte-identical.
    def no_policy_update(self, zs, task):
        return {}
    agent.update_pi = MethodType(no_policy_update, agent)
    observations = data["obs"].to(agent.device)
    actions = data["action"].to(agent.device)
    rewards = data["reward"].to(agent.device)
    task = torch.full((cfg.batch_size,), data["task"], device=agent.device, dtype=torch.long)
    history = []
    for update in range(updates):
        starts = torch.randint(len(actions) - cfg.horizon + 1, (cfg.batch_size,), device=agent.device)
        obs = torch.stack([observations[starts + t] for t in range(cfg.horizon + 1)])
        action = torch.stack([actions[starts + t] for t in range(cfg.horizon)])
        reward = torch.stack([rewards[starts + t] for t in range(cfg.horizon)]).unsqueeze(-1)
        terminated = torch.zeros_like(reward)
        info = agent._update(obs, action, reward, terminated, task)
        history.append({k: float(v) for k, v in info.items()})
    return history


def digest_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def same_state(before, module):
    after = module.state_dict()
    return all(torch.equal(before[k], after[k].detach().cpu()) for k in before)


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    conditions = [
        ("random", args.random_source),
        ("cascade", args.cascade_source),
        ("reverse_cascade", args.reverse_source),
        ("shuffled_cascade", args.shuffled_source),
    ]
    rows, comparisons = [], []
    for condition, source_path in conditions:
        agent, provenance = load_agent(args.checkpoint)
        if condition == "random":
            save(out / "provenance.json", provenance)
        data = torch.load(source_path, map_location="cpu", weights_only=False)
        before_non_q = digest_state(agent.model)
        before_actor = digest_state(agent.model._pi)
        history = adapt_q_only(agent, data, args.updates, args.learning_seed)
        if not same_state(before_actor, agent.model._pi):
            raise AssertionError("actor changed during q-only adaptation")
        # Every parameter outside _Qs must remain identical. Target Q buffers
        # are intentionally allowed to move through the upstream Polyak update.
        after_non_q = agent.model.state_dict()
        for name, value in before_non_q.items():
            if name.startswith("_Qs") or name.startswith("_target_Qs") or name.startswith("_detach_Qs"):
                continue
            current = after_non_q[name].detach().cpu()
            if not torch.equal(value, current):
                raise AssertionError(f"non-Q parameter changed: {name}")
        save(out / f"{condition}_adaptation.json", {"history": history, "actor_unchanged": True})
        validation, _ = episode(agent, args.target_task, args.validation_seed, policy="random", steps=args.eval_steps)
        # Dynamics-only validation uses the same held-out random trajectory for
        # every arm through the environment seed; the recorded value is a
        # model one-step latent MSE, independent of the Q head.
        obs_data = torch.load(source_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            task_id = agent.cfg.tasks.index(args.target_task)
            obs = obs_data["obs"].to(agent.device)
        result_row = {
            **validation,
            "condition": condition,
            "updates": args.updates,
            "actor_unchanged": True,
            "non_q_unchanged": True,
            "q_source": source_path,
        }
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row = {**result_row, **result, "seed": seed}
            rows.append(row)
            save(out / "results.json", rows)
            print(json.dumps(row), flush=True)
        save(out / f"{condition}_validation.json", {"random": validation})
    by = {(r["condition"], r["seed"]): r for r in rows}
    for seed in args.eval_seeds:
        rnd = by[("random", seed)]
        for condition in ["cascade", "reverse_cascade", "shuffled_cascade"]:
            x = by[(condition, seed)]
            comparisons.append({
                "seed": seed,
                "condition": condition,
                "mpc_return": x["return"],
                "random_return": rnd["return"],
                "loss_vs_random": 1 - x["return"] / max(rnd["return"], 1e-8),
            })
    save(out / "status.json", {
        "status": "completed",
        "results": rows,
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "decision": "Promote only if source-order-specific target MPC loss survives with byte-identical actor and non-Q world model; otherwise reject as ordinary shared-value transfer.",
    })
    print(json.dumps({"status": "completed", "comparisons": comparisons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--random-source", default="runs/tdmpc_order_controls_v0/random_source.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_order_controls_v0/cascade_source.pt")
    p.add_argument("--reverse-source", default="runs/tdmpc_order_controls_v0/reverse_cascade_source.pt")
    p.add_argument("--shuffled-source", default="runs/tdmpc_order_controls_v0/shuffled_cascade_source.pt")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--output", required=True)
    main(p.parse_args())
