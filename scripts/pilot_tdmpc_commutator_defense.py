"""Short native gate for a planner-specific temporal-commutator defense.

The defense adds a two-action latent composition penalty during the existing
dynamics-only consistency update. Replay order, observations, rewards, actor,
and the native MPC are unchanged. This is a falsification screen, not a claim
that the current cascade mechanism is caused by non-commutativity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


def adapt_commutator(agent, data, updates, seed, gamma):
    """Native dynamics update plus local two-action composition penalty."""
    set_seed(seed)
    cfg, model = agent.cfg, agent.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model._dynamics.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model._dynamics.parameters(), lr=cfg.lr)
    observations = data["obs"].to(agent.device)
    actions = data["action"].to(agent.device)
    task = data["task"]
    model.eval()
    with torch.no_grad():
        encoded = model.encode(observations, task)
    history = []
    for update in range(updates):
        starts = torch.randint(len(actions) - cfg.horizon + 1, (32,), device=agent.device)
        z = encoded[starts]
        consistency = torch.zeros((), device=agent.device)
        for step in range(cfg.horizon):
            z = model.next(z, actions[starts + step], task)
            consistency = consistency + cfg.rho ** step * F.mse_loss(z, encoded[starts + step + 1])
        consistency = consistency / cfg.horizon * cfg.consistency_coef

        # Compare T_b(T_a(z)) and T_a(T_b(z)) on the same local states. This
        # is differentiable through the dynamics parameters and uses no fake
        # target or reordered replay tuple.
        comm = torch.zeros((), device=agent.device)
        for step in range(cfg.horizon - 1):
            z0 = encoded[starts + step]
            a = actions[starts + step]
            b = actions[starts + step + 1]
            ab = model.next(model.next(z0, a, task), b, task)
            ba = model.next(model.next(z0, b, task), a, task)
            comm = comm + F.mse_loss(ab, ba)
        if cfg.horizon > 1:
            comm = comm / (cfg.horizon - 1)
        loss = consistency + gamma * comm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model._dynamics.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        history.append({"update": update + 1, "consistency": float(consistency.detach()),
                        "commutator": float(comm.detach()), "loss": float(loss.detach()),
                        "grad_norm": float(grad)})
    return history


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Native TD-MPC2 dynamics-only adaptation with differentiable two-action latent composition penalty; replay and planner untouched.",
        "gate": "Pass only if cascade return recovers >=80% of cascade-vs-random loss on both seeds, clean/no-source return is unchanged, and one-step error <=1.05x random.",
        "api_requests": 0, "gpu_work": "bounded native CUDA only",
    })
    started = time.monotonic()
    base, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    cascade = torch.load(args.cascade_source, weights_only=False)
    if len(cascade["action"]) != args.source_steps:
        raise ValueError("cascade source length differs from --source-steps")
    generator = torch.Generator().manual_seed(args.permutation_seed)
    permutations = {
        "reverse_cascade": torch.arange(args.source_steps - 1, -1, -1),
        "shuffled_cascade": torch.randperm(args.source_steps, generator=generator),
    }
    rows = []
    conditions = ["clean", "random", "cascade", "reverse_cascade", "shuffled_cascade"]
    for condition in conditions:
        agent, _ = load_agent(args.checkpoint)
        if condition == "clean":
            source = None
        elif condition == "random":
            source = collect(agent, args.source_task, args.source_seed, args.source_steps, "random")
        elif condition == "cascade":
            source = cascade
        else:
            source = replay_actions(agent, cascade["action"][permutations[condition]], args.source_task, args.source_seed)
        if source is not None:
            torch.save(source, out / f"{condition}_source.pt")
            before = frozen_digest(agent.model)
            history = adapt_commutator(agent, source, args.updates, args.learning_seed, args.gamma)
            if before != frozen_digest(agent.model):
                raise AssertionError("non-dynamics parameters changed")
            save(out / f"{condition}_adaptation.json", {"history": history})
        validation = collect(agent, args.target_task, args.validation_seed, args.eval_steps, "random")
        error = one_step_error(agent, validation, args.target_task)
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row = {**result, "condition": condition, "one_step_error": error,
                   "updates": 0 if source is None else args.updates, "gamma": args.gamma}
            rows.append(row)
            save(out / "results.json", rows)
            print(json.dumps(row), flush=True)
    by = {(r["condition"], r["seed"]): r for r in rows}
    gates = []
    for seed in args.eval_seeds:
        rnd = by[("random", seed)]["return"]
        cas = by[("cascade", seed)]["return"]
        rev = by[("reverse_cascade", seed)]["return"]
        shuf = by[("shuffled_cascade", seed)]["return"]
        # Recovery is measured against the unmitigated cascade loss from the
        # same run; reverse/shuffle are expected positive controls.
        clean = by[("clean", seed)]["return"]
        gates.append({"seed": seed, "random_return": rnd, "cascade_return": cas,
                      "reverse_return": rev, "shuffled_return": shuf,
                      "cascade_loss": 1.0 - cas / max(rnd, 1e-8),
                      "one_step_error_ratio": by[("cascade", seed)]["one_step_error"] /
                      max(by[("random", seed)]["one_step_error"], 1e-12),
                      "clean_return": clean,
                      "clean_vs_random_ratio": clean / max(rnd, 1e-8)})
    save(out / "status.json", {
        "status": "completed", "results": rows, "gates": gates,
        "elapsed_seconds": time.monotonic() - started,
        "decision": "Defense short gate only; promote only after clean retention and two-seed attack recovery with no order-control regression.",
    })
    print(json.dumps({"status": "completed", "gates": gates,
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
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--permutation-seed", type=int, default=933)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--output", required=True)
    main(p.parse_args())
