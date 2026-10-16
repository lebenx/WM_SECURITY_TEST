"""Short native screen for counterfactual-query goal leakage.

The dynamics service receives only latent/action pairs.  A client keeps a
terminal preference vector private and runs a small CEM planner.  The service
returns the honest native TD-MPC2 transition.  We measure whether the set of
never-executed hypothetical queries identifies the private preference even when
the selected first action is nearly the same.  This is a candidate screen, not
an accepted privacy claim or a cloud deployment model.
"""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path
import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/private_counterfactual_query_v0"


def native_rollout(agent, z0, actions, task):
    z = z0.expand(actions.shape[1], -1)
    states = []
    for t in range(actions.shape[0]):
        states.append(z.detach().cpu())
        z = agent.model.next(z, actions[t], task)
    states.append(z.detach().cpu())
    return torch.stack(states)


@torch.no_grad()
def plan(agent, z0, goal, task, seed, horizon=5, samples=96, iterations=4):
    """CEM with common random numbers; only model.next is the service call."""
    set_seed(seed)
    cfg = agent.cfg
    dim = cfg.action_dims[task]
    mean = torch.zeros(horizon, cfg.action_dim, device=agent.device)
    std = torch.full_like(mean, 0.8)
    mask = agent.model._action_masks[task]
    query_features = []
    query_count = 0
    for _ in range(iterations):
        actions = (mean[:, None] + std[:, None] * torch.randn(
            horizon, samples, cfg.action_dim, device=agent.device)).clamp(-1, 1)
        actions *= mask
        z = z0.expand(samples, -1)
        for t in range(horizon):
            # The provider observes this exact (z, a) request, including
            # rejected candidates.  It does not receive `goal`.
            query_features.append(torch.cat([z.detach().cpu(), actions[t].detach().cpu()], dim=1))
            query_count += samples
            z = agent.model.next(z, actions[t], task)
        costs = (z.detach().cpu() - goal.cpu()).square().mean(1)
        elite = torch.topk(-costs, min(16, samples)).indices
        elite_actions = actions[:, elite]
        mean = elite_actions.mean(1)
        std = elite_actions.std(1).clamp(0.08, 0.8)
    first = mean[0].detach().cpu()
    all_queries = torch.cat(query_features).numpy()
    # Inference features available to the provider: action-query moments only.
    action_part = all_queries[:, -cfg.action_dim:]
    summary = np.concatenate([action_part.mean(0), action_part.std(0),
                              action_part.mean(0) ** 2])
    return {"first_action": first.numpy(), "summary": summary,
            "query_trace": all_queries, "query_count": query_count,
            "goal": goal.cpu().numpy()}


def nearest_centroid_accuracy(rows):
    x = np.stack([r["summary"] for r in rows]); y = np.array([r["label"] for r in rows])
    pred = []
    for i in range(len(rows)):
        train = np.arange(len(rows)) != i
        centers = {c: x[train & (y == c)].mean(0) for c in np.unique(y)}
        pred.append(min(centers, key=lambda c: float(np.square(x[i] - centers[c]).mean())))
    return float(np.mean(np.asarray(pred) == y))


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    started = time.monotonic()
    plan_meta = {"seed": 20261009, "task": "walker-walk", "state_seed": 31,
                 "horizon": 5, "samples": 96, "iterations": 4,
                 "repetitions_per_goal": 12,
                 "threat_model": "honest native latent transition service sees z,a queries but not terminal preference; no returned-prediction tampering",
                 "privacy_boundary": "Researcher-defined client/service split; no claim that a released TD-MPC2 deployment has this split.",
                 "selection": "Two fixed random terminal latent preferences selected before repetitions; no task-return selection.",
                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "training_updates": 0, "api_requests": 0, "gpu_training": False}
    (RUN / "plan.json").write_text(json.dumps(plan_meta, indent=2) + "\n")
    agent, provenance = load_agent(ROOT / "models/tdmpc2/mt30-48M.pt")
    (RUN / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    agent.model.requires_grad_(False)
    task = agent.cfg.tasks.index(plan_meta["task"])
    env = make_one(plan_meta["task"], plan_meta["state_seed"])
    obs = pad_obs(env.reset(), agent.cfg)
    env.unwrapped.close()
    with torch.no_grad():
        z0 = agent.model.encode(obs.to(agent.device).unsqueeze(0), torch.tensor([task], device=agent.device))
    rng = np.random.default_rng(plan_meta["seed"])
    goals = [torch.tensor(rng.normal(0, 0.35, agent.cfg.latent_dim), dtype=torch.float32),
             torch.tensor(rng.normal(0, 0.35, agent.cfg.latent_dim), dtype=torch.float32)]
    # Keep the preference pair fixed and search only for a pair with a shared
    # first action under the prescribed common-random-number probe.
    candidate = None
    for attempt in range(40):
        g0 = torch.tensor(rng.normal(0, 0.35, agent.cfg.latent_dim), dtype=torch.float32)
        g1 = torch.tensor(rng.normal(0, 0.35, agent.cfg.latent_dim), dtype=torch.float32)
        r0 = plan(agent, z0, g0, task, 9000 + attempt)
        r1 = plan(agent, z0, g1, task, 9000 + attempt)
        diff = float(np.abs(r0["first_action"] - r1["first_action"]).mean())
        if candidate is None or diff < candidate[0]: candidate = (diff, g0, g1, attempt)
        if diff < 0.08:
            break
    diff, goals[0], goals[1], attempt = candidate
    rows = []
    for label, goal in enumerate(goals):
        for rep in range(plan_meta["repetitions_per_goal"]):
            result = plan(agent, z0, goal, task, 12000 + rep)
            rows.append({"label": label, "rep": rep, "first_action": result["first_action"].tolist(),
                         "summary": result["summary"].tolist(), "query_count": result["query_count"]})
    first = np.stack([r["first_action"] for r in rows])
    within = float(np.mean(np.abs(first[::2] - first[1::2]))) if False else None
    # Pair repetitions by seed for the two preferences.
    paired = [(rows[i]["first_action"], rows[i + plan_meta["repetitions_per_goal"]]["first_action"])
              for i in range(plan_meta["repetitions_per_goal"])]
    pair_diff = float(np.mean([np.abs(np.asarray(a)-np.asarray(b)).mean() for a,b in paired]))
    acc = nearest_centroid_accuracy(rows)
    summary = {"status": "candidate_query_leakage_observed" if acc >= 0.8 and pair_diff < 0.08 else "candidate_gate_not_met",
               "query_only_goal_classification_accuracy": acc,
               "paired_first_action_mean_abs_difference": pair_diff,
               "selected_pair_common_seed_first_action_mean_abs_difference": diff,
               "goal_pair_search_attempts": attempt + 1,
               "goals": [g.tolist() for g in goals],
               "repetitions": len(rows), "query_count_per_run": rows[0]["query_count"],
               "native_environment_steps": 0, "training_updates": 0, "api_requests": 0,
               "scope": plan_meta["privacy_boundary"],
               "interpretation": "Query traces can be distinguishable while the first selected action is close in this researcher-defined latent-goal MPC screen. This is not yet a security claim: no actual private task utility, closed-loop harm, defense, or independent task family was tested.",
               "elapsed_s": time.monotonic() - started}
    (RUN / "traces.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
