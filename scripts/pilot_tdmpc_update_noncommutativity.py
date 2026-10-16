"""Measure non-commuting dynamics-update operators on legal source data.

For matched source windows A and B, estimate the first-order difference
between applying a small native dynamics gradient step A then B versus B then
A.  This is a mechanism diagnostic for the proposed temporal-order imprinting;
it is not evidence of a security attack by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wm_security.tdmpc_native import load_agent


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def gradients(agent, data, start):
    cfg, model = agent.cfg, agent.model
    params = [p for p in model._dynamics.parameters() if p.requires_grad]
    obs = data["obs"].to(agent.device)
    actions = data["action"].to(agent.device)
    task = data["task"]
    starts = torch.as_tensor(start, device=agent.device, dtype=torch.long)
    with torch.no_grad():
        encoded = model.encode(obs, task)
    z = encoded[starts]
    loss = torch.zeros((), device=agent.device)
    for t in range(cfg.horizon):
        z = model.next(z, actions[starts + t], task)
        loss = loss + cfg.rho ** t * F.mse_loss(z, encoded[starts + t + 1])
    loss = loss / cfg.horizon
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(params, gs)], float(loss.detach().cpu())


def add_step(params, grads, scale):
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(g, alpha=scale)


def norm(grads):
    return math.sqrt(sum(float(g.square().sum().cpu()) for g in grads))


def pair(agent, data, a, b, step_size):
    params = [p for p in agent.model._dynamics.parameters() if p.requires_grad]
    ga, loss_a = gradients(agent, data, a)
    add_step(params, ga, -step_size)
    gb_after_a, _ = gradients(agent, data, b)
    add_step(params, ga, step_size)
    gb, loss_b = gradients(agent, data, b)
    add_step(params, gb, -step_size)
    ga_after_b, _ = gradients(agent, data, a)
    add_step(params, gb, step_size)
    cross = [((gba - gb) - (gab - ga)) for gba, gb, gab, ga in zip(gb_after_a, gb, ga_after_b, ga)]
    denom = norm(ga) + norm(gb) + 1e-12
    return {"loss_a": loss_a, "loss_b": loss_b,
            "grad_a_norm": norm(ga), "grad_b_norm": norm(gb),
            "cross_difference_norm": norm(cross),
            "normalized_cross_difference": norm(cross) / denom}


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Native TD-MPC2 dynamics gradients on identical source-window indices; finite small-step A/B versus B/A update commutator.",
        "conditions": ["random", "cascade", "reverse_cascade", "shuffled_cascade"],
        "interpretation": "A structural diagnostic. It does not establish task harm or novelty unless cascade has a selective excess and the matching controls agree.",
        "api_requests": 0,
    })
    started = time.monotonic()
    source_paths = {
        "random": args.random_source,
        "cascade": args.cascade_source,
        "reverse_cascade": args.reverse_source,
        "shuffled_cascade": args.shuffled_source,
    }
    rows = []
    for condition, path in source_paths.items():
        agent, provenance = load_agent(args.checkpoint)
        if not rows:
            save(out / "provenance.json", provenance)
        data = torch.load(path, weights_only=False)
        for j in range(args.pairs):
            # Same starts for all conditions; pair separation is fixed.
            a = [int((2*j*args.pair_stride + k) % (len(data["action"]) - agent.cfg.horizon)) for k in range(args.batch)]
            b = [int((2*j*args.pair_stride + args.pair_gap + k) % (len(data["action"]) - agent.cfg.horizon)) for k in range(args.batch)]
            row = pair(agent, data, a, b, args.step_size)
            row.update({"condition": condition, "pair": j})
            rows.append(row)
            save(out / "results.json", rows)
        print(json.dumps({"condition": condition,
                          "mean_normalized_cross_difference": float(sum(r["normalized_cross_difference"] for r in rows if r["condition"] == condition) / args.pairs)}), flush=True)
    summary = {}
    for condition in source_paths:
        vals = [r["normalized_cross_difference"] for r in rows if r["condition"] == condition]
        summary[condition] = {"mean": float(sum(vals) / len(vals)), "std": float(torch.tensor(vals).std()), "n": len(vals)}
    save(out / "status.json", {"status": "completed", "summary": summary, "rows": rows,
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Retain only as mechanism evidence; no broad claim follows from noncommutativity alone."})
    print(json.dumps({"status": "completed", "summary": summary,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--random-source", default="runs/tdmpc_order_controls_v0/random_source.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_order_controls_v0/cascade_source.pt")
    p.add_argument("--reverse-source", default="runs/tdmpc_order_controls_v0/reverse_cascade_source.pt")
    p.add_argument("--shuffled-source", default="runs/tdmpc_order_controls_v0/shuffled_cascade_source.pt")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--pair-stride", type=int, default=19)
    p.add_argument("--pair-gap", type=int, default=7)
    p.add_argument("--step-size", type=float, default=0.01)
    p.add_argument("--output", required=True)
    main(p.parse_args())
