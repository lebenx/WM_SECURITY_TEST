"""Measure horizon-specific latent drift after temporal-order adaptation.

This follows the strict missing experiment from the claim review.  It does not
assume non-normality: each condition is compared on the same held-out
walker-run action sequence, with one-step error and open-loop latent drift for
multiple horizons.  The first-action Jacobian is an auxiliary diagnostic.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect
from pilot_tdmpc_order_controls import replay_actions


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def rollout_drift(agent, validation, task_name, horizons, starts):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    obs = validation["obs"].to(agent.device)
    acts = validation["action"].to(agent.device)
    with torch.no_grad():
        encoded = model.encode(obs, task)
    rows = []
    for h in horizons:
        mse, norm = [], []
        for i in starts:
            if i + h >= len(acts):
                continue
            with torch.no_grad():
                z = encoded[i:i+1]
                for k in range(h):
                    z = model.next(z, acts[i+k:i+k+1], task)
                err = z[0] - encoded[i+h]
                mse.append(float(err.square().mean().cpu()))
                norm.append(float(err.norm().cpu()))
        rows.append({"horizon": h, "latent_mse": float(np.mean(mse)),
                     "latent_l2": float(np.mean(norm)), "n": len(mse)})
    # A small, reproducible first-action sensitivity calculation.  It is not
    # a claim about spectral radius; it measures the planning-relevant map.
    jac_rows = []
    for i in starts[: min(8, len(starts))]:
        for h in horizons:
            if i + h >= len(acts):
                continue
            with torch.no_grad():
                z0 = encoded[i:i+1]
                tail = acts[i+1:i+h] if h > 1 else acts[i+1:i+1]
            a0 = acts[i:i+1].detach().clone().requires_grad_(True)
            def f(a):
                z = model.next(z0, a, task)
                for aa in tail:
                    z = model.next(z, aa.unsqueeze(0), task)
                return z[0]
            with torch.enable_grad():
                jac = torch.autograd.functional.jacobian(f, a0, create_graph=False)
            jac_rows.append({"start": int(i), "horizon": h,
                             "jacobian_fro": float(jac.norm().detach().cpu())})
    for h in horizons:
        vals = [r["jacobian_fro"] for r in jac_rows if r["horizon"] == h]
        next_row = next(r for r in rows if r["horizon"] == h)
        next_row["first_action_jacobian_fro"] = float(np.mean(vals)) if vals else float("nan")
    return rows


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Fixed held-out walker-run random trajectory; compare clean, random, cascade, reverse, and shuffled source adaptations on multi-horizon latent rollout drift.",
        "interpretation": "Diagnostic for temporal-order-specific planning amplification; no non-normality or broad security claim is inferred from drift alone.",
        "api_requests": 0,
    })
    started = time.monotonic()
    base, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    validation = collect(base, args.target_task, args.validation_seed, args.eval_steps, "random")
    cascade = torch.load(args.cascade_source, weights_only=False)
    gen = torch.Generator().manual_seed(args.permutation_seed)
    perms = {
        "reverse_cascade": torch.arange(args.source_steps - 1, -1, -1),
        "shuffled_cascade": torch.randperm(args.source_steps, generator=gen),
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
            adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        horizon_rows = rollout_drift(agent, validation, args.target_task, args.horizons,
                                     list(range(0, args.eval_steps - max(args.horizons), args.start_stride)))
        rows.append({"condition": condition, "horizon_rows": horizon_rows})
        save(out / "results.json", rows)
        print(json.dumps(rows[-1]), flush=True)
    save(out / "status.json", {"status": "completed", "conditions": rows,
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Use as mechanism evidence only; causal promotion still requires an intervention that selectively restores the cascade horizon curve."})
    print(json.dumps({"status": "completed", "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_cascade_heldout_matched_v1/cascade_source.pt")
    p.add_argument("--source-task", default="cheetah-run")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--source-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--permutation-seed", type=int, default=933)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--start-stride", type=int, default=8)
    p.add_argument("--output", required=True)
    main(p.parse_args())
