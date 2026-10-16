"""Joint-phase order controls for the legal cascade source.

The action envelope and each time step's action norm are preserved while the
six actuator coordinates are permuted before truthful simulator replay.  This
tests whether the strongest source effect depends on a directional
cross-actuator phase ordering rather than merely high amplitude or sample
count.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import episode, load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error
from pilot_tdmpc_order_controls import replay_actions


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Permute actuator coordinates of the same legal cascade action envelope, replay each in the simulator, and adapt only native TD-MPC2 dynamics.",
        "conditions": ["random", "cascade", "joint_reverse", "joint_rotate"],
        "gate": "A cross-actuator phase-order mechanism requires cascade harm against random and both coordinate controls, with one-step error <=1.05x random.",
        "api_requests": 0,
    })
    started = time.monotonic()
    base, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    cascade = torch.load(args.cascade_source, weights_only=False)
    dim = base.cfg.action_dims[base.cfg.tasks.index(args.source_task)]
    actions = cascade["action"].clone()
    reverse = actions.clone(); reverse[:, :dim] = actions[:, :dim].flip(dims=[1])
    rotate = actions.clone(); rotate[:, :dim] = torch.roll(actions[:, :dim], shifts=1, dims=1)
    variants = {"cascade": cascade,
                "joint_reverse": replay_actions(base, reverse, args.source_task, args.source_seed),
                "joint_rotate": replay_actions(base, rotate, args.source_task, args.source_seed)}
    rows = []
    for condition in ["clean", "random", "cascade", "joint_reverse", "joint_rotate"]:
        agent, _ = load_agent(args.checkpoint)
        if condition == "clean":
            source = None
        elif condition == "random":
            source = collect(agent, args.source_task, args.source_seed, args.source_steps, "random")
        else:
            source = variants[condition]
        if source is not None:
            torch.save(source, out / f"{condition}_source.pt")
            adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        validation = collect(agent, args.target_task, args.validation_seed, args.eval_steps, "random")
        error = one_step_error(agent, validation, args.target_task)
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row = {**result, "condition": condition, "one_step_error": error,
                   "source_steps": 0 if source is None else len(source["action"]),
                   "action_rms": None if source is None else float(source["action"][:, :dim].square().mean().sqrt())}
            rows.append(row); save(out / "results.json", rows); print(json.dumps(row), flush=True)
    by = {(r["condition"], r["seed"]): r for r in rows}
    comparisons = []
    for seed in args.eval_seeds:
        rnd = by[("random", seed)]
        for condition in ["cascade", "joint_reverse", "joint_rotate"]:
            x = by[(condition, seed)]
            comparisons.append({"seed": seed, "condition": condition,
                                "return": x["return"], "random_return": rnd["return"],
                                "loss_vs_random": 1.0 - x["return"] / max(rnd["return"], 1e-8),
                                "error_ratio_vs_random": x["one_step_error"] / max(rnd["one_step_error"], 1e-12),
                                "action_rms_ratio_vs_cascade": x["action_rms"] / max(by[("cascade", seed)]["action_rms"], 1e-12)})
    save(out / "status.json", {"status": "completed", "results": rows, "comparisons": comparisons,
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Short phase-order diagnostic; require independent source seeds and a planner-ranking intervention before promotion."})
    print(json.dumps({"status": "completed", "comparisons": comparisons,
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
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--output", required=True)
    main(p.parse_args())
