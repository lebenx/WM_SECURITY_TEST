"""Short diagnostic for padded-action shadowing in multitask TD-MPC2.

The official mt30 model pads a low-dimensional task's legal action vector into
the shared maximum action dimension.  This screen asks whether native,
task-labeled adaptation on a two-dimensional reacher task can selectively
change the four action directions that were always padded with zero, while
leaving a held-out walker trajectory's one-step latent error unchanged.

This is a mechanism diagnostic, not an attack result: the source data and
updates are researcher-selected, the model is offline, and the existing native
walker returns are used only as a task-harm control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from wm_security.tdmpc_native import load_agent
from pilot_tdmpc_adaptation import adapt, frozen_digest


ROOT = Path(__file__).resolve().parents[1]


def save(out: Path, name: str, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@torch.no_grad()
def latent_error(agent, data, task):
    obs = data["obs"].to(agent.device)
    action = data["action"].to(agent.device)
    task_t = torch.full((obs.shape[0] - 1,), task, device=agent.device, dtype=torch.long)
    z = agent.model.encode(obs[:-1], task_t)
    target = agent.model.encode(obs[1:], task_t)
    pred = agent.model.next(z, action, task_t)
    return float(torch.mean((pred - target) ** 2))


@torch.no_grad()
def finite_action_sensitivity(agent, data, task, dims, count=64, epsilon=0.05):
    obs = data["obs"][:count].to(agent.device)
    base = data["action"][:count].to(agent.device)
    task_t = torch.full((obs.shape[0],), task, device=agent.device, dtype=torch.long)
    z = agent.model.encode(obs, task_t)
    plus, minus = base.clone(), base.clone()
    plus[:, dims] += epsilon
    minus[:, dims] -= epsilon
    delta = agent.model.next(z, plus, task_t) - agent.model.next(z, minus, task_t)
    return float(torch.linalg.vector_norm(delta, dim=-1).mean() / (2 * epsilon))


def main():
    out = ROOT / "runs/tdmpc_padded_action_shadowing_v0"
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "models/tdmpc2/mt30-48M.pt"
    source_dir = ROOT / "runs/tdmpc_adaptation_screen_v0"
    source_files = {"mpc": source_dir / "mpc_data.pt", "random": source_dir / "random_data.pt"}
    victim_file = source_dir / "clean_walker-walk.pt"
    plan = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_task": "reacher-easy",
        "victim_task": "walker-walk",
        "source_action_dims": 2,
        "shared_action_dim": 6,
        "padded_dims": [2, 3, 4, 5],
        "source_files": {k: str(v.relative_to(ROOT)) for k, v in source_files.items()},
        "victim_file": str(victim_file.relative_to(ROOT)),
        "updates": 40,
        "scope": "Native task-labeled dynamics-only adaptation on completed reacher data; no label forgery, observation edit, target search, or deployment claim.",
        "diagnostic": "Compare victim on-trajectory one-step latent MSE and finite action sensitivity for first two versus padded four action dimensions before/after adaptation.",
        "controls": ["clean", "native_mpc_source", "native_random_source"],
        "api_requests": 0,
        "gpu_training": True,
    }
    save(out, "plan.json", plan)
    victim = torch.load(victim_file, map_location="cpu", weights_only=False)
    # The source task's unused action coordinates must be exactly zero for the
    # operation to be interpretable as legal padding rather than action edits.
    source_tail = {}
    for name, path in source_files.items():
        d = torch.load(path, map_location="cpu", weights_only=False)
        source_tail[name] = {
            "task_id": int(d["task"]),
            "tail_max_abs": float(d["action"][:, 2:].abs().max()),
            "active_max_abs": float(d["action"][:, :2].abs().max()),
        }
    if any(x["tail_max_abs"] != 0.0 for x in source_tail.values()):
        raise AssertionError("Source data is not exactly padded in unused action dimensions.")
    save(out, "source_padding_audit.json", source_tail)

    results = []
    started = time.monotonic()
    clean_agent, provenance = load_agent(checkpoint)
    save(out, "provenance.json", provenance)
    clean_hash = frozen_digest(clean_agent.model)
    clean_mse = latent_error(clean_agent, victim, task=1)
    clean_active = finite_action_sensitivity(clean_agent, victim, 1, [0, 1])
    clean_padded = finite_action_sensitivity(clean_agent, victim, 1, [2, 3, 4, 5])
    results.append({
        "condition": "clean",
        "victim_one_step_latent_mse": clean_mse,
        "active_dims_sensitivity": clean_active,
        "padded_dims_sensitivity": clean_padded,
        "model_digest": clean_hash,
    })
    for name, path in source_files.items():
        agent, _ = load_agent(checkpoint)
        source = torch.load(path, map_location="cpu", weights_only=False)
        before = frozen_digest(agent.model)
        history = adapt(agent, source, 40, 31, "dynamics")
        after = frozen_digest(agent.model)
        if after != before:
            raise AssertionError("Non-dynamics parameters changed during adaptation.")
        row = {
            "condition": f"native_{name}_source",
            "victim_one_step_latent_mse": latent_error(agent, victim, 1),
            "active_dims_sensitivity": finite_action_sensitivity(agent, victim, 1, [0, 1]),
            "padded_dims_sensitivity": finite_action_sensitivity(agent, victim, 1, [2, 3, 4, 5]),
            "model_digest": before,
            "non_dynamics_digest_unchanged": before == after,
            "updates": len(history),
            "final_consistency_loss": history[-1]["weighted_consistency_loss"],
        }
        results.append(row)
        save(out, f"{name}_adaptation.json", {"history": history, "model_digest": before})
    base = results[0]
    for row in results[1:]:
        row["padded_relative_change"] = row["padded_dims_sensitivity"] / max(base["padded_dims_sensitivity"], 1e-12) - 1.0
        row["active_relative_change"] = row["active_dims_sensitivity"] / max(base["active_dims_sensitivity"], 1e-12) - 1.0
        row["victim_mse_ratio"] = row["victim_one_step_latent_mse"] / max(base["victim_one_step_latent_mse"], 1e-12)
    summary = {
        "status": "completed",
        "results": results,
        "source_padding_exact": True,
        "native_walker_task_harm_control": {
            "existing_clean_return": 972.4091031141579,
            "existing_mpc_reacher_source_return": 972.2975431717932,
            "existing_random_reacher_source_return": 973.7530029825866,
            "source": "runs/tdmpc_adaptation_screen_v0/results.json",
        },
        "decision": "Diagnostic only. A candidate would require a selective padded-dimension change with held-out on-policy preservation, replicated native walker loss, and a legal deployed learning threat model. This run does not establish those conditions or novelty.",
        "elapsed_seconds": time.monotonic() - started,
        "api_requests": 0,
        "gpu_training": True,
    }
    save(out, "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
