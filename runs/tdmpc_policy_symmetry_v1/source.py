"""Native short screen of policy-preserving action reflection in dynamics.

This is a researcher-installed dynamics-input wrapper, not data poisoning or a
checkpoint-only attack. In real arithmetic, reflecting the pre-tanh Gaussian
about its conditional mean preserves the native stochastic prior's action law.
Thus state-only rollouts under that prior, and expected additive returns with
unchanged rewards, agree. Action-labelled transitions do NOT agree. Numerical
preservation is measured separately; finite precision is not an exact theorem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import types

import torch

from wm_security.tdmpc_native import episode, load_agent, set_seed
from pilot_tdmpc_query_interference import model_digest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/tdmpc_policy_symmetry_v1"
BASE = ROOT / "runs/tdmpc_dynamics_reliance_v0"
TASKS = ["walker-walk", "reacher-easy", "cup-catch"]
SEED = 31


def save(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_parameters(model, z, task):
    # Same deterministic calculations as the upstream pi(); draw no randomness.
    from common import math as native_math
    mean, raw_std = model._pi(model.task_emb(z, task)).chunk(2, dim=-1)
    mask = model._action_masks[task]
    log_std = native_math.log_std(raw_std, model.log_std_min, model.log_std_dif)
    return mean * mask, log_std * mask, mask


def reflect(action, mean):
    # The continuous extension maps +1 to -1 and -1 to +1. No clipping of
    # planner actions or arbitrary epsilon at the endpoints is introduced.
    return torch.tanh(2 * mean.double() - torch.atanh(action.double())).to(action.dtype)


@torch.no_grad()
def inspect_preservation(agent, task):
    idx = agent.cfg.tasks.index(task)
    task_id = torch.tensor([idx], device=agent.device)
    trace = torch.load(BASE / f"{task}_native_mpc.pt", weights_only=True)
    z = agent.model.encode(trace["obs"][:-1].to(agent.device), task_id)
    a = trace["action"].to(agent.device)
    target = agent.model.encode(trace["obs"][1:].to(agent.device), task_id)
    mean, _, mask = prior_parameters(agent.model, z, task_id)
    clean = agent.model.next(z, a, task_id)
    changed = agent.model.next(z, reflect(a, mean) * mask, task_id)
    negated = agent.model.next(z, -a, task_id)
    restored = agent.model.next(z, reflect(reflect(a, mean), mean) * mask, task_id)
    mse = lambda pred: float((pred.double() - target.double()).square().mean())
    results = {
        "real_action_labelled_transitions": len(a),
        "latent_prediction_mse": {"clean": mse(clean), "prior_reflection": mse(changed),
                                 "sign_flip": mse(negated)},
        "double_reflection_prediction_max_abs": float((restored - clean).abs().max()),
    }
    # 32 predetermined anchors, 128 independent innovations plus their exact
    # opposites. Compare transformed samples to the paired native samples.
    # This is an algebraic/numerical control, NOT an independent two-sample test.
    selected = torch.linspace(0, len(z) - 1, 32).long().to(agent.device)
    anchor = z[selected]
    m, log_std, mask = prior_parameters(agent.model, anchor, task_id)
    set_seed(20261009 + idx)
    eps = torch.randn(32, 128, agent.cfg.action_dim, device=agent.device)
    eps = torch.cat([eps, -eps], dim=1) * mask
    sampled = torch.tanh(m[:, None] + log_std[:, None].exp() * eps) * mask
    pair_order = torch.cat([torch.arange(128, 256), torch.arange(128)]).to(agent.device)
    reflected = reflect(sampled, m[:, None]) * mask
    paired = sampled[:, pair_order]
    zr = anchor[:, None].expand(-1, 256, -1).reshape(-1, z.shape[-1])
    before_rng = torch.cuda.get_rng_state().clone()
    remean, _, _ = prior_parameters(agent.model, zr, task_id)
    _ = reflect(sampled.reshape(-1, agent.cfg.action_dim), remean)
    rng_unchanged = torch.equal(before_rng, torch.cuda.get_rng_state())
    assert rng_unchanged
    native_pred = agent.model.next(zr, paired.reshape(-1, agent.cfg.action_dim), task_id)
    ref_pred = agent.model.next(zr, reflected.reshape(-1, agent.cfg.action_dim), task_id)
    raw_pred = agent.model.next(zr, -sampled.reshape(-1, agent.cfg.action_dim), task_id)
    active = mask.bool().expand_as(sampled)
    results.update(
        antithetic_anchors=32, prior_actions_per_anchor=256,
        numerical_prior_endpoint_fraction=float((sampled[active].abs() == 1).float().mean()),
        reflected_vs_paired_action_max_abs=float((reflected - paired).abs().max()),
        reflected_vs_paired_prediction_max_abs=float((ref_pred - native_pred).abs().max()),
        reflected_vs_paired_prediction_rmse=float((ref_pred.double() - native_pred.double()).square().mean().sqrt()),
        sign_flip_vs_paired_prediction_rmse=float((raw_pred.double() - native_pred.double()).square().mean().sqrt()),
        deterministic_wrapper_preserves_cuda_rng=rng_unchanged,
    )
    torch.save({"sampled_actions": sampled.cpu(), "reflected_actions": reflected.cpu(),
                "paired_actions": paired.cpu(), "native_paired_predictions": native_pred.cpu(),
                "reflected_predictions": ref_pred.cpu(), "sign_flip_predictions": raw_pred.cpu(),
                "labelled_clean": clean.cpu(), "labelled_reflection": changed.cpu(),
                "labelled_sign_flip": negated.cpu(), "labelled_target": target.cpu()},
               RUN / f"{task}_preservation.pt")
    return results


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    baseline = json.loads((BASE / "summary.json").read_text())
    save("plan.json", {
        "tasks": TASKS, "seeds": [SEED], "steps_per_episode": 500,
        "arms": ["prior_reflection", "sign_flip", "double_reflection"],
        "reference": str(BASE.relative_to(ROOT)), "reference_summary_sha256": sha(BASE / "summary.json"),
        "script_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(ROOT / "src/wm_security/tdmpc_native.py"),
        "reference_trace_sha256": {t: sha(BASE / f"{t}_native_mpc.pt") for t in TASKS},
        "operation": "Only model.next receives phi_z(a)=tanh(2*mu(z)-atanh(a)); reward/Q/encoder/prior/physical action unchanged. Deterministic native pre-tanh prior mean, no RNG draw. All proposal/evaluation dynamics calls use the wrapper.",
        "control": "Raw sign reversal tests ordinary action inversion; applying phi twice restores the original dynamics up to measured numerical error.",
        "preservation": "Real-arithmetic state-only trajectory law under the FIXED native stochastic prior; expected additive return also preserved with unchanged reward. Not the action-labelled trajectory law, not reward trajectory law, not all policies, and not exact finite-precision equivalence.",
        "development_task_gate": "Continue this candidate only if prior_reflection reduces return by at least200/1000 on at least2 tasks. This gate alone is insufficient to accept a novel or stealthy security claim.",
        "numerical_gate": "Before any new episode, require antithetic action error<=1e-4, prediction RMSE<=1e-5, and double-reflection prediction max error<=1e-4. These are screening tolerances, not an exact finite-precision distribution proof.",
        "previous_attempt": "v0 stopped after numerical preservation failed: one completed walker reflection and an interrupted sign-flip comparator. Reuse its walker inspection; test the other2 native tasks without episodes unless this new gate passes. Do not count v0 harm as preservation evidence.",
        "novelty_boundary": "Policy-specific model/value equivalence is prior art (Grimm et al., NeurIPS2020); action permutation and bad dynamics are not by themselves novel. This pilot measures the concrete native effect and limitations.",
        "privilege": "Researcher-installed runtime dynamics-input wrapper with white-box access to native prior mean; no demonstrated data-poisoning or checkpoint-only realization.",
        "training_updates": 0, "api_requests": 0, "gpu_work": "Short frozen inference only.",
    })
    agent, provenance = load_agent(ROOT / "models/tdmpc2/mt30-48M.pt")
    save("provenance.json", provenance)
    agent.model.requires_grad_(False)
    with torch.no_grad():
        agent.model._task_emb(torch.arange(len(agent.cfg.tasks), device=agent.device))
    before = model_digest(agent.model)
    assert before == baseline["model_digest"]
    assert provenance["planner"] == json.loads((BASE / "provenance.json").read_text())["planner"]
    original = agent.model.next
    inspections, rows = {}, []
    counts = {"calls": 0, "transitions": 0}
    gates = {}

    for task in TASKS:
        if task == "walker-walk":
            old = ROOT / "runs/tdmpc_policy_symmetry_v0/preservation.json"
            inspections[task] = json.loads(old.read_text())[task]
            inspections[task]["reused_from"] = str(old.relative_to(ROOT))
            inspections[task]["reused_source_sha256"] = sha(old)
        else:
            inspections[task] = inspect_preservation(agent, task)
        save("preservation.json", inspections)
        print(json.dumps({"phase": "preservation", "task": task, **inspections[task]}), flush=True)
        p = inspections[task]
        gates[task] = (p["reflected_vs_paired_action_max_abs"] <= 1e-4 and
                       p["reflected_vs_paired_prediction_rmse"] <= 1e-5 and
                       p["double_reflection_prediction_max_abs"] <= 1e-4)
        if not gates[task]:
            print(json.dumps({"phase": "skip_episodes", "task": task,
                              "reason": "numerical_preservation_gate_failed"}), flush=True)
            continue
        for condition in ["prior_reflection", "sign_flip", "double_reflection"]:
            counts.update(calls=0, transitions=0)
            def changed_next(model, z, a, task_id):
                counts["calls"] += 1
                counts["transitions"] += z.shape[0]
                if condition == "sign_flip":
                    mapped = -a
                else:
                    mean, _, mask = prior_parameters(model, z, task_id)
                    mapped = reflect(a, mean) * mask
                    if condition == "double_reflection":
                        mapped = reflect(mapped, mean) * mask
                return original(z, mapped, task_id)
            agent.model.next = types.MethodType(changed_next, agent.model)
            agent._prev_mean.zero_()
            print(json.dumps({"phase": "episode", "task": task, "condition": condition}), flush=True)
            try:
                row, trace = episode(agent, task, SEED, "mpc", 500,
                                     lambda p: print(json.dumps(dict(p, task=task, condition=condition)), flush=True))
            finally:
                delattr(agent.model, "next")
            assert model_digest(agent.model) == before
            reference_trace = torch.load(BASE / f"{task}_native_mpc.pt", weights_only=True)
            assert torch.equal(trace["obs"][0], reference_trace["obs"][0])
            row.update(condition=condition, dynamics_calls=counts["calls"],
                       dynamics_transitions=counts["transitions"], model_weights_unchanged=True)
            rows.append(row)
            torch.save(trace, RUN / f"{task}_{condition}.pt")
            save("results.json", rows)
            print(json.dumps(row), flush=True)
    effects = []
    for task in TASKS:
        reference = next(r for r in baseline["effects"] if r["task"] == task)
        returns = {r["condition"]: r["return"] for r in rows if r["task"] == task}
        if not returns:
            continue
        effects.append({"task": task, "native_return": reference["native_return"],
                        "native_prior_return": reference["native_prior_return"], **returns,
                        "reflection_drop": reference["native_return"] - returns["prior_reflection"]})
    gate = sum(e["reflection_drop"] >= 200 for e in effects) >= 2
    save("summary.json", {
        "status": "completed", "effects": effects, "development_task_gate_passed": gate,
        "numerical_preservation_gates": gates,
        "preservation": inspections, "results": rows, "model_digest": before,
        "model_weights_unchanged": True, "new_native_environment_steps": sum(r["steps"] for r in rows),
        "new_episodes": len(rows), "training_updates": 0, "api_requests": 0,
        "elapsed_s": time.monotonic() - started,
        "accepted_security_claim": None,
        "scope": "Single paired development seed in3tasks; runtime dynamics-input intervention. Reference controls reused from prior fixed run with same model/planner/seed. State-law argument is specific to fixed stochastic prior and omits action labels; empirical paired check is not an independent distribution test. No stealth or novelty established.",
    })
    print(json.dumps({"status": "completed", "effects": effects, "gate": gate}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if RUN.exists():
            save("failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
