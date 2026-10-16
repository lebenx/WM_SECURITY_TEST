"""Full native-update baseline for the temporal-order source controls."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from wm_security.tdmpc_native import episode, load_agent
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect, one_step_error


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Same legal source trajectories as order-control gate, but use upstream native full update including encoder, reward/value, task embedding and policy.",
        "conditions": ["random", "cascade", "reverse_cascade", "shuffled_cascade"],
        "gate": "A full-update replication is supportive only if cascade retains selective walker-run MPC loss, actor/one-step controls remain interpretable, and cup-catch remains negative.",
        "api_requests": 0,
    })
    started = time.monotonic()
    rows = []
    for condition, source_path in [("random", args.random_source), ("cascade", args.cascade_source),
                                   ("reverse_cascade", args.reverse_source), ("shuffled_cascade", args.shuffled_source)]:
        agent, provenance = load_agent(args.checkpoint)
        if not rows:
            save(out / "provenance.json", provenance)
        source = torch.load(source_path, weights_only=False)
        history = adapt(agent, source, args.updates, args.learning_seed, "native_full")
        save(out / f"{condition}_adaptation.json", {"history": history})
        for task in args.targets:
            validation = collect(agent, task, args.validation_seed, args.eval_steps, "random")
            error = one_step_error(agent, validation, task)
            for seed in args.eval_seeds:
                for policy in ["mpc", "prior"]:
                    result, _ = episode(agent, task, seed, policy=policy, steps=args.eval_steps)
                    row = {**result, "condition": condition, "one_step_error": error,
                           "update_scope": "native_full", "updates": args.updates}
                    rows.append(row); save(out / "results.json", rows); print(json.dumps(row), flush=True)
    by = {(r["condition"], r["task"], r["policy"], r["seed"]): r for r in rows}
    comparisons = []
    for task in args.targets:
        for seed in args.eval_seeds:
            rnd = by[("random", task, "mpc", seed)]
            for condition in ["cascade", "reverse_cascade", "shuffled_cascade"]:
                x = by[(condition, task, "mpc", seed)]
                comparisons.append({"task": task, "seed": seed, "condition": condition,
                                    "mpc_return": x["return"], "random_return": rnd["return"],
                                    "loss_vs_random": 1.0 - x["return"] / max(rnd["return"], 1e-8),
                                    "error_ratio_vs_random": x["one_step_error"] / max(rnd["one_step_error"], 1e-12),
                                    "actor_return": by[(condition, task, "prior", seed)]["return"],
                                    "random_actor_return": by[("random", task, "prior", seed)]["return"]})
    save(out / "status.json", {"status": "completed", "results": rows, "comparisons": comparisons,
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Full-update strong baseline only; do not promote without independent replication."})
    print(json.dumps({"status": "completed", "comparisons": comparisons,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--random-source", default="runs/tdmpc_order_controls_v0/random_source.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_order_controls_v0/cascade_source.pt")
    p.add_argument("--reverse-source", default="runs/tdmpc_order_controls_v0/reverse_cascade_source.pt")
    p.add_argument("--shuffled-source", default="runs/tdmpc_order_controls_v0/shuffled_cascade_source.pt")
    p.add_argument("--targets", nargs="+", default=["walker-run", "cup-catch"])
    p.add_argument("--validation-seed", type=int, default=33)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--output", required=True)
    main(p.parse_args())
