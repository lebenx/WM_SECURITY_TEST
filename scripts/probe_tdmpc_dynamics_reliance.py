"""Short native control ablation: does action-conditioned dynamics matter?

Uses the published controller unchanged. The ablation calls its original
dynamics with zero actions; encoder, rewards, critic, prior and real actions
are unchanged. This is a causal baseline, not a novel or stealthy attack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import types

import torch

from wm_security.tdmpc_native import episode, load_agent
from pilot_tdmpc_query_interference import model_digest


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/tdmpc_dynamics_reliance_v0"
TASKS = ["walker-walk", "reacher-easy", "cup-catch"]
SEED = 31


def save(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    save("plan.json", {
        "tasks": TASKS, "seeds": [SEED], "steps_per_episode": 500,
        "conditions": ["native_mpc", "action_blind_dynamics", "native_policy_prior"],
        "clean_gate": "Native MPC return >=900/1000 per task; skip the other arms for a task that fails.",
        "action_blind_dynamics": "Every native model.next(z,a,task) call instead invokes the SAME learned next(z,zeros_like(a),task), including policy proposal rollouts. Physical actions, reward/Q/prior/encoder inputs and all model weights remain unchanged.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((ROOT / "src/wm_security/tdmpc_native.py").read_bytes()).hexdigest(),
        "checkpoint": "models/tdmpc2/mt30-48M.pt",
        "randomness": "Native episode helper resets Python/NumPy/CPU/CUDA seeds per arm. Reset warm start before each episode; same simulator seed.",
        "scope": "One paired development seed in three native tasks. Causal necessity baseline only; no novel security claim, stealth/fidelity constraint, trained attack, or population effect estimate.",
        "purpose": "Select competent native tasks with measured action-conditioned dynamics dependence before constructing another world-model security mechanism.",
        "training_updates": 0, "api_requests": 0, "gpu_work": "Frozen short inference only.",
    })
    agent, provenance = load_agent(ROOT / "models/tdmpc2/mt30-48M.pt")
    save("provenance.json", provenance)
    agent.model.requires_grad_(False)
    # Embedding(max_norm=1) can perform its native renormalisation on access.
    # Do this equally for all arms before taking the frozen baseline digest.
    with torch.no_grad():
        old_embedding = agent.model._task_emb.weight.detach().clone()
        agent.model._task_emb(torch.arange(len(agent.cfg.tasks), device=agent.device))
        normalization_max_delta = float((old_embedding - agent.model._task_emb.weight).abs().max())
    before = model_digest(agent.model)
    original_next = agent.model.next
    rows = []
    counts = {"calls": 0, "transitions": 0}

    def action_blind(model, z, a, task):
        counts["calls"] += 1
        counts["transitions"] += int(z.shape[0])
        return original_next(z, torch.zeros_like(a), task)

    for task in TASKS:
        for condition in ["native_mpc", "action_blind_dynamics", "native_policy_prior"]:
            counts.update(calls=0, transitions=0)
            agent._prev_mean.zero_()
            if condition == "action_blind_dynamics":
                agent.model.next = types.MethodType(action_blind, agent.model)
            print(json.dumps({"phase": "start", "task": task, "condition": condition}), flush=True)
            try:
                row, trace = episode(
                    agent, task, SEED, "prior" if condition == "native_policy_prior" else "mpc",
                    steps=500, progress=lambda r: print(json.dumps(dict(r, task=task, condition=condition)), flush=True))
            finally:
                # Removing the local override restores the class's native method.
                if "next" in agent.model.__dict__:
                    delattr(agent.model, "next")
            assert model_digest(agent.model) == before
            row.update(condition=condition, model_weights_unchanged=True,
                       ablated_dynamics_calls=counts["calls"], ablated_transitions=counts["transitions"])
            rows.append(row)
            torch.save(trace, RUN / f"{task}_{condition}.pt")
            save("results.json", rows)
            print(json.dumps(row), flush=True)
            if condition == "native_mpc" and row["return"] < 900:
                break
    effects = []
    for task in TASKS:
        by = {r["condition"]: r for r in rows if r["task"] == task}
        base = by["native_mpc"]["return"]
        e = {"task": task, "native_return": base, "clean_gate_passed": base >= 900}
        if "action_blind_dynamics" in by:
            e.update(action_blind_return=by["action_blind_dynamics"]["return"],
                     action_blind_return_drop=base - by["action_blind_dynamics"]["return"],
                     native_prior_return=by["native_policy_prior"]["return"],
                     mpc_advantage_over_prior=base - by["native_policy_prior"]["return"])
        effects.append(e)
    save("summary.json", {
        "status": "completed", "effects": effects, "results": rows,
        "model_digest": before, "model_weights_unchanged": True,
        "native_embedding_normalization_max_delta": normalization_max_delta,
        "native_environment_steps": sum(r["steps"] for r in rows),
        "ablated_dynamics_calls": sum(r["ablated_dynamics_calls"] for r in rows),
        "ablated_transitions": sum(r["ablated_transitions"] for r in rows),
        "training_updates": 0, "api_requests": 0,
        "elapsed_s": time.monotonic() - start,
        "interpretation": "Task-level causal ablation of action conditioning, not an attack or novel mechanism. A high clean score alone does not measure dynamics reliance. Results are one seed per task; strong-baseline findings do not establish ICLR-scale evidence.",
    })
    print(json.dumps({"status": "completed", "effects": effects}), flush=True)


if __name__ == "__main__":
    main()
