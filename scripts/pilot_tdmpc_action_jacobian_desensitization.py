"""Short native gate for orthogonal-task action-Jacobian desensitization."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import episode, load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import one_step_error


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def collect_source(agent, task_name, seed, steps, mode):
    set_seed(seed)
    idx = agent.cfg.tasks.index(task_name)
    env = make_one(task_name, seed)
    observations, actions, rewards = [pad_obs(env.reset(), agent.cfg)], [], []
    try:
        for t in range(steps):
            dim = agent.cfg.action_dims[idx]
            action = torch.zeros(agent.cfg.action_dim)
            if mode == "dither":
                # Equal RMS high-frequency legal actuator excitation.
                phase = torch.arange(dim, dtype=torch.float32) * (math.pi / max(dim, 1))
                action[:dim] = (0.72 * torch.sin(2 * math.pi * t / 4 + phase)).clamp(-1, 1)
            elif mode == "random":
                action[:dim] = torch.rand(dim) * 2 - 1
            else:
                raise ValueError(mode)
            nxt, reward, done, _ = env.step(action[:dim])
            observations.append(pad_obs(nxt, agent.cfg)); actions.append(action)
            rewards.append(float(reward))
            if done:
                break
    finally:
        env.unwrapped.close()
    return {"obs": torch.stack(observations), "action": torch.stack(actions),
            "reward": torch.tensor(rewards), "task": idx, "seed": seed}


def action_jacobian(agent, validation, task_name, count=4):
    model = agent.model; task = agent.cfg.tasks.index(task_name)
    obs = validation["obs"].to(agent.device); acts = validation["action"].to(agent.device)
    ids = torch.linspace(0, len(acts) - 1, count).long().tolist()
    vals = []
    with torch.no_grad():
        zs = [model.encode(obs[i:i+1], task)[0] for i in ids]
    for z, i in zip(zs, ids):
        a = acts[i].detach().clone().requires_grad_(True)
        def f(x): return model.next(z.unsqueeze(0), x.unsqueeze(0), task)[0]
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f, a, create_graph=False)
        sv = torch.linalg.svdvals(jac)
        vals.append({"fro": float(jac.norm().detach().cpu()),
                     "largest_sv": float(sv.max().detach().cpu()),
                     "effective_rank": int((sv > sv.max() * 1e-3).sum().detach().cpu())})
    return {"rows": vals, "mean_fro": float(np.mean([x["fro"] for x in vals])),
            "mean_largest_sv": float(np.mean([x["largest_sv"] for x in vals])),
            "mean_effective_rank": float(np.mean([x["effective_rank"] for x in vals]))}


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Legal high-frequency walker-stand dither versus equal-budget random source; native TD-MPC2 dynamics-only update; evaluate orthogonal reacher-easy action Jacobian and MPC.",
        "falsification": "Require dither reach action-Jacobian Frobenius norm <=0.60x random, one-step error <=1.10x random, and MPC return <=0.50x random; otherwise reject C2.",
        "api_requests": 0,
        "gpu_work": "short native CUDA only",
    })
    started = time.monotonic(); rows = []
    base, provenance = load_agent(args.checkpoint); save(out / "provenance.json", provenance)
    for condition in ["clean", "random", "dither"]:
        agent, _ = load_agent(args.checkpoint)
        if condition == "clean":
            source = None
        else:
            source = collect_source(agent, args.source_task, args.source_seed, args.source_steps, condition)
            torch.save(source, out / f"{condition}_source.pt")
            adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        validation = collect_source(agent, args.target_task, args.validation_seed, args.eval_steps, "random")
        jac = action_jacobian(agent, validation, args.target_task, args.jacobian_states)
        error = one_step_error(agent, validation, args.target_task)
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row = {"condition": condition, "seed": seed, "return": result["return"],
                   "one_step_error": error, "action_jacobian": jac,
                   "updates": 0 if source is None else args.updates}
            rows.append(row); save(out / "results.json", rows); print(json.dumps(row), flush=True)
    by = {(r["condition"], r["seed"]): r for r in rows}
    gates = []
    for seed in args.eval_seeds:
        rnd, dith = by[("random", seed)], by[("dither", seed)]
        gates.append({"seed": seed,
                      "return_ratio": dith["return"] / max(rnd["return"], 1e-8),
                      "jacobian_fro_ratio": dith["action_jacobian"]["mean_fro"] / max(rnd["action_jacobian"]["mean_fro"], 1e-12),
                      "error_ratio": dith["one_step_error"] / max(rnd["one_step_error"], 1e-12),
                      "passed": (dith["return"] <= 0.50 * rnd["return"] and
                                 dith["action_jacobian"]["mean_fro"] <= 0.60 * rnd["action_jacobian"]["mean_fro"] and
                                 dith["one_step_error"] <= 1.10 * rnd["one_step_error"])})
    save(out / "status.json", {"status": "completed", "results": rows, "gates": gates,
                                "all_passed": all(x["passed"] for x in gates),
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Short C2 gate only; scale only if all seed gates pass and SimNorm/unnormalized controls isolate the contraction."})
    print(json.dumps({"status": "completed", "gates": gates,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--source-task", default="walker-stand")
    p.add_argument("--target-task", default="reacher-easy")
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--source-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--jacobian-states", type=int, default=4)
    p.add_argument("--output", required=True)
    main(p.parse_args())
