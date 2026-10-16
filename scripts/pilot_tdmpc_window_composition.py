"""Test whether ordered source effects require multi-step training windows.

The source transitions and update dose are held fixed.  Only the consistency
window used by the frozen-dynamics adaptation changes from the native horizon
3 to one-step training.  If cascade harm disappears while actor and one-step
fit remain controlled, this supports a trajectory co-occurrence/composition
mediator rather than a generic source distribution shift.
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


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    rows = []
    conditions = [
        ("random", args.random_source),
        ("cascade", args.cascade_source),
        ("reverse_cascade", args.reverse_source),
        ("shuffled_cascade", args.shuffled_source),
    ]
    for condition, source_path in conditions:
        agent, provenance = load_agent(args.checkpoint)
        if condition == "random":
            save(out / "provenance.json", provenance)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        old_horizon = agent.cfg.horizon
        agent.cfg.horizon = args.train_horizon
        history = adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        agent.cfg.horizon = old_horizon
        # Same held-out random target trajectory is collected after every arm;
        # model evaluation uses native horizon 3 and native MPC unchanged.
        validation = collect(agent, args.target_task, args.validation_seed,
                             args.eval_steps, "random")
        err = one_step_error(agent, validation, args.target_task)
        for seed in args.eval_seeds:
            result, _ = episode(agent, args.target_task, seed, policy="mpc", steps=args.eval_steps)
            row = {**result, "condition": condition, "train_horizon": args.train_horizon,
                   "updates": args.updates, "one_step_error": err,
                   "actor_unchanged": True, "source_path": source_path}
            rows.append(row)
            save(out / "results.json", rows)
            print(json.dumps(row), flush=True)
        save(out / f"{condition}_adaptation.json", {"history": history, "train_horizon": args.train_horizon})
    by = {(r["condition"], r["seed"]): r for r in rows}
    comparisons = []
    for seed in args.eval_seeds:
        rnd = by[("random", seed)]
        for condition in ["cascade", "reverse_cascade", "shuffled_cascade"]:
            x = by[(condition, seed)]
            comparisons.append({"seed": seed, "condition": condition,
                                "mpc_return": x["return"], "random_return": rnd["return"],
                                "loss_vs_random": 1 - x["return"] / max(rnd["return"], 1e-8),
                                "error_ratio_vs_random": x["one_step_error"] / max(rnd["one_step_error"], 1e-12)})
    save(out / "status.json", {"status": "completed", "results": rows,
                                "comparisons": comparisons,
                                "elapsed_seconds": time.monotonic() - start,
                                "decision": "Promote window-composition mechanism only if native-horizon training shows selective cascade loss and one-step training removes it without changing the deployment planner."})
    print(json.dumps({"status": "completed", "comparisons": comparisons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--random-source", default="runs/tdmpc_order_controls_v0/random_source.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_order_controls_v0/cascade_source.pt")
    p.add_argument("--reverse-source", default="runs/tdmpc_order_controls_v0/reverse_cascade_source.pt")
    p.add_argument("--shuffled-source", default="runs/tdmpc_order_controls_v0/shuffled_cascade_source.pt")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--train-horizon", type=int, default=1)
    p.add_argument("--output", required=True)
    main(p.parse_args())
