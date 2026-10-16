"""Short gate for Cross-morphology Latent Contraction Invariant (CLCI).

The proposed mediator predicts that authentic source updates align the source
and target action-sensitivity subspaces in shared latent dynamics, while
compressing target counterfactual branch dispersion.  This diagnostic keeps
the native dynamics-only update and MPC controls unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import episode, load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error
from pilot_tdmpc_order_controls import replay_actions


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def action_jacobian(agent, obs, task_name, action):
    model, cfg = agent.model, agent.cfg
    task = cfg.tasks.index(task_name)
    with torch.no_grad():
        z = model.encode(obs.to(agent.device).unsqueeze(0), task)[0].detach()
    a0 = action.to(agent.device).detach().clone().requires_grad_(True)

    def f(a):
        return model.next(z.unsqueeze(0), a.unsqueeze(0), task)[0]

    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(f, a0, create_graph=False)
    return J.detach()


def subspace_score(v1, v2, k):
    # v matrices are action_dim x k right-singular subspaces.
    m = v1[:, :k].T @ v2[:, :k]
    return float(torch.linalg.svdvals(m).sum().cpu() / k)


def jacobian_metrics(agent, data, task_name, count, k):
    n = len(data["action"])
    idxs = torch.linspace(0, max(0, n - 1), count).long().tolist()
    rows, subspaces = [], []
    for i in idxs:
        action = data["action"][i]
        J = action_jacobian(agent, data["obs"][i], task_name, action)
        u, s, vh = torch.linalg.svd(J, full_matrices=False)
        subspaces.append(vh.T[:, :k])
        rows.append({"index": int(i), "singular_values": s.detach().cpu().tolist(),
                     "frobenius": float(torch.linalg.matrix_norm(J).cpu()),
                     "top_singular": float(s[0].cpu())})
    score = 0.0
    if len(subspaces) >= 2:
        score = float(np.mean([subspace_score(subspaces[i], subspaces[j], k)
                               for i in range(len(subspaces))
                               for j in range(i + 1, len(subspaces))]))
    return {"rows": rows, "within_task_topk_alignment": score,
            "mean_top_singular": float(np.mean([r["top_singular"] for r in rows]))}


def rollout_dispersion(agent, task_name, obs, horizon, population, seed):
    cfg, model = agent.cfg, agent.model
    task = cfg.tasks.index(task_name)
    g = torch.Generator(device=agent.device).manual_seed(seed)
    actions = torch.rand((horizon, population, cfg.action_dim), device=agent.device,
                         generator=g) * 2 - 1
    actions *= model._action_masks[task]
    with torch.no_grad():
        z = model.encode(obs.to(agent.device).unsqueeze(0), task).repeat(population, 1)
        for t in range(horizon):
            z = model.next(z, actions[t], task)
        # Mean pairwise distance over a bounded random subset.
        d = torch.pdist(z[: min(population, 64)].float())
    return float(d.mean().cpu()) if d.numel() else 0.0


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    base, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    cascade = torch.load(args.cascade_source, map_location="cpu", weights_only=False)
    rows = []
    for condition in ["random", "cascade", "reverse_cascade", "shuffled_cascade"]:
        agent, _ = load_agent(args.checkpoint)
        if condition == "random":
            source = collect(agent, args.source_task, args.source_seed, args.source_steps, "random")
        elif condition == "cascade":
            source = cascade
        else:
            perm = (torch.arange(args.source_steps - 1, -1, -1)
                    if condition == "reverse_cascade" else
                    torch.randperm(args.source_steps, generator=torch.Generator().manual_seed(args.permutation_seed)))
            source = replay_actions(agent, cascade["action"][perm], args.source_task, args.source_seed)
        adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        target = collect(agent, args.target_task, args.validation_seed, args.eval_steps, "random")
        source_metrics = jacobian_metrics(agent, source, args.source_task, args.jacobian_states, args.top_k)
        target_metrics = jacobian_metrics(agent, target, args.target_task, args.jacobian_states, args.top_k)
        # Compare source and target subspaces at matched index locations.
        source_sub = []
        target_sub = []
        for i in torch.linspace(0, len(source["action"]) - 1, args.jacobian_states).long().tolist():
            J = action_jacobian(agent, source["obs"][i], args.source_task, source["action"][i])
            source_sub.append(torch.linalg.svd(J, full_matrices=False).Vh.T[:, :args.top_k])
        for i in torch.linspace(0, len(target["action"]) - 1, args.jacobian_states).long().tolist():
            J = action_jacobian(agent, target["obs"][i], args.target_task, target["action"][i])
            target_sub.append(torch.linalg.svd(J, full_matrices=False).Vh.T[:, :args.top_k])
        cross_alignment = float(np.mean([subspace_score(a, b, args.top_k)
                                         for a in source_sub for b in target_sub]))
        row = {
            "condition": condition,
            "source_action_subspace": source_metrics,
            "target_action_subspace": target_metrics,
            "source_target_topk_alignment": cross_alignment,
            "target_branch_dispersion_h3": rollout_dispersion(agent, args.target_task,
                                                               target["obs"][0], 3, args.population,
                                                               args.branch_seed),
            "one_step_error": one_step_error(agent, target, args.target_task),
            "updates": args.updates,
        }
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row[f"return_seed_{seed}"] = result["return"]
        rows.append(row)
        save(out / "results.json", rows)
        print(json.dumps(row), flush=True)
    save(out / "status.json", {"status": "completed", "rows": rows,
                                "elapsed_seconds": time.monotonic() - start,
                                "decision": "Promote CLCI only if cascade has selective low branch dispersion and high source-target action-subspace alignment while retaining low one-step error and return loss across both seeds."})
    print(json.dumps({"status": "completed", "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_order_controls_v0/cascade_source.pt")
    p.add_argument("--source-task", default="cheetah-run")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--source-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--permutation-seed", type=int, default=933)
    p.add_argument("--jacobian-states", type=int, default=3)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--branch-seed", type=int, default=20261010)
    p.add_argument("--output", required=True)
    main(p.parse_args())
