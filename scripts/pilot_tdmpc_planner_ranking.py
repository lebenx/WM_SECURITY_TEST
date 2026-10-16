"""Measure whether temporal-order adaptation changes MPC candidate ranking.

At identical walker-run initial states, every adapted model receives the same
candidate action sequences.  We compare the model's finite-horizon value
ranking with the true simulator return of each sequence.  This is a direct
planning-specific diagnostic and does not infer a broad attack from rank
correlation alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm_security.tdmpc_native import load_agent, make_one, pad_obs, set_seed
from pilot_tdmpc_adaptation import adapt
from screen_tdmpc_non_normal import collect
from pilot_tdmpc_order_controls import replay_actions


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def rank_corr(x, y):
    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        out = np.empty_like(order, dtype=float)
        out[order] = np.arange(len(a), dtype=float)
        return out
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def true_returns(task, seed, bank):
    idx = []
    for sequence in bank:
        env = make_one(task, seed)
        try:
            env.reset()
            total = 0.0
            for action in sequence:
                _, reward, done, _ = env.step(torch.as_tensor(action, dtype=torch.float32))
                total += float(reward)
                if done:
                    break
        finally:
            env.unwrapped.close()
        idx.append(total)
    return np.asarray(idx, dtype=float)


def model_values(agent, task, seed, bank):
    set_seed(seed)
    env = make_one(task, seed)
    try:
        obs = pad_obs(env.reset(), agent.cfg)
    finally:
        env.unwrapped.close()
    task_id = agent.cfg.tasks.index(task)
    with torch.no_grad():
        z = agent.model.encode(obs.to(agent.device).unsqueeze(0), task_id)
        actions = torch.as_tensor(np.asarray(bank), dtype=torch.float32, device=agent.device)
        # _estimate_value expects [horizon, population, action_dim].
        actions = actions.transpose(0, 1)
        # The upstream helper allocates its termination tensor from
        # cfg.num_samples.  This diagnostic intentionally uses a smaller
        # fixed bank, so keep the helper's internal population in sync with
        # the actual candidate batch instead of relying on the planner's
        # default sample count.
        agent.cfg.num_samples = int(actions.shape[1])
        z = z.repeat(actions.shape[1], 1)
        values = agent._estimate_value(z, actions, torch.tensor([task_id], device=agent.device))
    return values.squeeze(-1).detach().cpu().numpy()


def main(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {
        **vars(args),
        "operation": "Fixed candidate action bank at identical walker-run roots; compare native model finite-horizon value ranking with true simulator ranking after temporal-order source adaptation.",
        "conditions": ["clean", "random", "cascade", "reverse_cascade", "shuffled_cascade"],
        "interpretation": "A ranking-specific diagnostic. It is evidence for planner amplification only if cascade rank fidelity drops while reverse/shuffle controls do not; it is not by itself an ICLR claim.",
        "api_requests": 0,
    })
    started = time.monotonic()
    base, provenance = load_agent(args.checkpoint)
    save(out / "provenance.json", provenance)
    cascade = torch.load(args.cascade_source, weights_only=False)
    gen = torch.Generator().manual_seed(args.bank_seed)
    # Keep the bank legal and identical across all model conditions.
    bank = (torch.rand((args.population, args.horizon, base.cfg.action_dim), generator=gen) * 2 - 1).numpy()
    target_dim = base.cfg.action_dims[base.cfg.tasks.index(args.target_task)]
    bank[:, :, target_dim:] = 0.0
    true_by_seed = {seed: true_returns(args.target_task, seed, bank[:, :, :target_dim])
                    for seed in args.eval_seeds}
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
            perm = (torch.arange(args.source_steps - 1, -1, -1)
                    if condition == "reverse_cascade"
                    else torch.randperm(args.source_steps, generator=torch.Generator().manual_seed(args.permutation_seed)))
            source = replay_actions(agent, cascade["action"][perm], args.source_task, args.source_seed)
        if source is not None:
            adapt(agent, source, args.updates, args.learning_seed, "dynamics")
        for seed in args.eval_seeds:
            predicted = model_values(agent, args.target_task, seed, bank)
            actual = true_by_seed[seed]
            rows.append({"condition": condition, "seed": seed,
                         "spearman": rank_corr(predicted, actual),
                         "predicted_top10_true_mean": float(actual[np.argsort(predicted)[-10:]].mean()),
                         "oracle_top10_true_mean": float(actual[np.argsort(actual)[-10:]].mean()),
                         "predicted_value_mean": float(np.mean(predicted)),
                         "true_return_mean": float(np.mean(actual))})
            save(out / "results.json", rows)
            print(json.dumps(rows[-1]), flush=True)
    save(out / "status.json", {"status": "completed", "results": rows,
                                "elapsed_seconds": time.monotonic() - started,
                                "decision": "Use rank fidelity to decide whether a selective planning-order mechanism remains plausible; do not infer causality from this diagnostic alone."})
    print(json.dumps({"status": "completed", "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/tdmpc2/mt30-48M.pt")
    p.add_argument("--cascade-source", default="runs/tdmpc_cascade_heldout_matched_v1/cascade_source.pt")
    p.add_argument("--source-task", default="cheetah-run")
    p.add_argument("--target-task", default="walker-run")
    p.add_argument("--source-seed", type=int, default=23)
    p.add_argument("--source-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--learning-seed", type=int, default=31)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[31, 32])
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--bank-seed", type=int, default=2026)
    p.add_argument("--permutation-seed", type=int, default=933)
    p.add_argument("--output", required=True)
    main(p.parse_args())
