"""Bounded source/target matrix for the legal cascade planning-harm screen.

This is a discovery gate, not a final benchmark.  Each source arm uses the
same legal monotone actuator curriculum and a matched random trajectory,
updates only the native TD-MPC2 dynamics head, and evaluates unchanged native
MPC on several held-out tasks/seeds.  The matrix is deliberately small so a
positive pattern can be followed by a preregistered replication before any
long run.
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
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        **vars(args),
        "operation": "Legal monotone actuator cascade versus matched random source trajectories; native TD-MPC2 dynamics-only consistency adaptation; unchanged native MPC evaluation.",
        "controls": ["clean checkpoint", "random source trajectory", "cascade source trajectory"],
        "promotion": "This matrix is exploratory. A broad claim requires a preregistered source/target family replication, low one-step-error gate, shuffled-order control, and a useful defense.",
        "api_requests": 0,
        "gpu_work": "bounded native CUDA adaptation and evaluation",
    }
    save(out / "plan.json", plan)
    started = time.monotonic()
    results = []
    clean = {}
    for target in args.targets:
        for seed in args.eval_seeds:
            agent, provenance = load_agent(args.checkpoint)
            if not clean:
                save(out / "provenance.json", provenance)
            row, _ = episode(agent, target, seed, policy="mpc", steps=args.eval_steps)
            row.update({"condition": "clean", "source_task": None, "target_task": target,
                        "eval_seed": seed, "one_step_error": None, "updates": 0})
            clean[(target, seed)] = row
            results.append(row)
            print(json.dumps(row), flush=True)
    for source_task in args.sources:
        for condition in ("random", "cascade"):
            agent, provenance = load_agent(args.checkpoint)
            source = collect(agent, source_task, args.source_seed, args.source_steps, condition,
                             cascade_gain=args.cascade_gain)
            torch.save(source, out / f"{source_task.replace('-', '_')}_{condition}_source.pt")
            history = adapt(agent, source, args.updates, seed=args.learning_seed, scope="dynamics")
            save(out / f"{source_task.replace('-', '_')}_{condition}_adaptation.json",
                 {"history": history, "source_task": source_task, "condition": condition})
            for target in args.targets:
                for seed in args.eval_seeds:
                    row, _ = episode(agent, target, seed, policy="mpc", steps=args.eval_steps)
                    validation = collect(agent, target, args.validation_seed, args.eval_steps, "random")
                    row.update({"condition": condition, "source_task": source_task,
                                "target_task": target, "eval_seed": seed,
                                "one_step_error": one_step_error(agent, validation, target),
                                "updates": args.updates, "source_steps": len(source["action"])})
                    results.append(row)
                    print(json.dumps(row), flush=True)
            save(out / "results.json", results)
    # Summarize only after all raw rows are persisted.
    by = {(r["condition"], r.get("source_task"), r["target_task"], r["eval_seed"]): r
          for r in results}
    comparisons = []
    for source_task in args.sources:
        for target in args.targets:
            for seed in args.eval_seeds:
                rnd = by[("random", source_task, target, seed)]
                cas = by[("cascade", source_task, target, seed)]
                comparisons.append({
                    "source_task": source_task, "target_task": target, "eval_seed": seed,
                    "clean_return": clean[(target, seed)]["return"],
                    "random_return": rnd["return"], "cascade_return": cas["return"],
                    "cascade_loss_vs_random": 1.0 - cas["return"] / max(rnd["return"], 1e-8),
                    "cascade_error_ratio_vs_random": cas["one_step_error"] / max(rnd["one_step_error"], 1e-12),
                    "low_error_gate": cas["one_step_error"] <= 1.05 * rnd["one_step_error"],
                })
    status = {
        "status": "completed",
        "results": results,
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "decision": "Exploratory matrix only; do not promote a broad mechanism from this search. Require independent replication and causal intervention before scaling.",
    }
    save(out / "status.json", status)
    print(json.dumps({"status": status["status"], "comparisons": comparisons,
                      "elapsed_seconds": status["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--sources", nargs="+", default=["cheetah-run", "walker-walk-backwards", "walker-walk", "hopper-hop"])
    p.add_argument("--targets", nargs="+", default=["walker-run", "walker-walk", "walker-stand", "hopper-hop"])
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--source-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--cascade-gain", type=float, default=1.25)
    p.add_argument("--output", required=True)
    main(p.parse_args())
