"""Native short gate for recurrent belief carryover across episode boundaries.

This is a bounded carrier experiment, not a claim.  A GRU world model is
trained on real MBRL-Lib CartPole transitions while observing only position and
pole angle; velocity is latent and must be carried by recurrent context.  A
source client first supplies a legal trajectory, then a target episode starts
without clearing the model belief.  The victim uses unchanged CEM planning.
The clean and reset branches clear the recurrent state at the target boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mbrl.env.cartpole_continuous import CartPoleEnv


ROOT = Path(__file__).resolve().parents[1]
OBS_DIM = 2  # x and pole angle; velocities are hidden from the learned model.
ACT_DIM = 1
PRED_DIM = OBS_DIM + 1  # next partial observation delta and reward.


class RecurrentWorldModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(OBS_DIM + ACT_DIM, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, PRED_DIM))

    def encode(self, obs, act, h=None):
        x = torch.cat([obs, act], dim=-1)
        y, h = self.gru(x, h)
        return y, h

    def predict_from_hidden(self, obs, act, h):
        y, h2 = self.encode(obs, act, h)
        return self.head(y), h2


def partial(obs):
    return np.asarray(obs, dtype=np.float32)[..., [0, 2]]


def collect(seed, episodes=160, max_steps=50):
    rng = np.random.default_rng(seed)
    env = CartPoleEnv()
    sequences = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rows = []
        for _ in range(max_steps):
            # Mix random and mild stabilizing actions so the dataset contains
            # both failure boundaries and competent near-equilibrium motion.
            if rng.random() < 0.5:
                act = env.action_space.sample()
            else:
                act = np.asarray([-np.sign(obs[2] + 0.4 * obs[3])], dtype=np.float32)
                act += rng.normal(0, 0.15, size=1).astype(np.float32)
                act = np.clip(act, -1, 1)
            nxt, reward, done, truncated, _ = env.step(act)
            rows.append((partial(obs), act.astype(np.float32), partial(nxt), float(reward)))
            obs = nxt
            if done or truncated:
                break
        if len(rows) >= 8:
            sequences.append(rows)
    env.close()
    return sequences


def sequence_tensors(rows, device="cpu"):
    obs = torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32, device=device)
    act = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32, device=device)
    nxt = torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.float32, device=device)
    rew = torch.tensor(np.asarray([r[3] for r in rows], dtype=np.float32)[:, None], device=device)
    return obs, act, torch.cat([nxt - obs, rew], dim=-1)


def train_model(model, sequences, epochs=80, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 1000)
    losses = []
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(sequences))
        total = 0.0
        for idx in order:
            obs, act, target = sequence_tensors(sequences[idx])
            pred, _ = model.encode(obs[None], act[None])
            pred = model.head(pred[0])
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
        losses.append(total / len(order))
    model.eval()
    return losses


def rollout_hidden(model, rows, h=None):
    obs, act, _ = sequence_tensors(rows)
    with torch.no_grad():
        _, h = model.encode(obs[None], act[None], h)
    return h.detach()


def one_step_rmse(model, rows, h=None):
    obs, act, target = sequence_tensors(rows)
    with torch.no_grad():
        pred, _ = model.predict_from_hidden(obs[None], act[None], h)
    return float((pred[0] - target).square().mean().sqrt())


def model_rollout(model, obs0, h0, actions):
    """Predict rewards and advance hidden state for one candidate sequence."""
    obs = obs0
    h = h0
    rewards = []
    with torch.no_grad():
        for act in actions:
            act_t = torch.as_tensor(act, dtype=torch.float32).reshape(1, 1, 1)
            obs_t = obs.reshape(1, 1, OBS_DIM)
            pred, h = model.predict_from_hidden(obs_t, act_t, h)
            nxt_delta = pred[0, 0, :OBS_DIM]
            obs = obs + nxt_delta
            # CartPole's native termination boundary is part of the planning
            # interface.  The compact model does not have a separate terminal
            # head, so use the known physical bounds as a smooth survival
            # penalty; this gives CEM a competent short-horizon controller
            # without exposing hidden velocities to the model.
            angle_excess = torch.relu(obs[1].abs() - 0.209)
            pos_excess = torch.relu(obs[0].abs() - 2.4)
            rewards.append(pred[0, 0, OBS_DIM] - 25.0 * angle_excess - 2.0 * pos_excess)
    return torch.stack(rewards).sum(), h


def cem_action(model, obs, h, rng, horizon=8, population=64, iterations=2, elite=8):
    mean = np.zeros(horizon, dtype=np.float32)
    std = np.ones(horizon, dtype=np.float32)
    for _ in range(iterations):
        actions = np.clip(rng.normal(mean, std, size=(population, horizon)), -1, 1).astype(np.float32)
        vals = []
        for row in actions:
            value, _ = model_rollout(model, obs, h, row)
            vals.append(float(value))
        vals = np.asarray(vals)
        ids = np.argsort(vals)[-elite:]
        mean = actions[ids].mean(0)
        std = np.maximum(actions[ids].std(0), 0.1)
    return np.asarray([mean[0]], dtype=np.float32)


def source_rows(env, seed, steps):
    obs, _ = env.reset(seed=seed)
    rows = []
    for i in range(steps):
        act = np.asarray([np.sin(0.37 * i)], dtype=np.float32)
        nxt, reward, done, truncated, _ = env.step(act)
        rows.append((partial(obs), act, partial(nxt), float(reward)))
        obs = nxt
        if done or truncated:
            break
    return rows


def evaluate(model, seed, condition, episodes=2, max_steps=80, source_steps=20,
             cem_horizon=8, cem_population=64, cem_iterations=2, cem_elite=8):
    rng = np.random.default_rng(seed + 30000)
    returns = []
    lengths = []
    target_rmses = []
    for ep in range(episodes):
        env = CartPoleEnv()
        source = source_rows(env, seed + 7000 + ep, source_steps)
        target_seed = seed + ep
        target_obs, _ = env.reset(seed=target_seed)
        if condition == "carryover":
            h = rollout_hidden(model, source)
        elif condition in {"clean", "reset_defense"}:
            h = None
        else:
            raise ValueError(condition)
        obs = partial(target_obs)
        total = 0.0
        n = 0
        for step in range(max_steps):
            action = cem_action(model, torch.tensor(obs, dtype=torch.float32), h, rng,
                                horizon=cem_horizon, population=cem_population,
                                iterations=cem_iterations, elite=cem_elite)
            # Standard recurrent dynamics interface: one current observation
            # and the selected current action produce one next observation and
            # advance the belief exactly once.  The previous version fed the
            # current observation through a neutral action before planning and
            # then fed it a second time, making the clean controller invalid.
            obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, 1, OBS_DIM)
            act_t = torch.tensor(action, dtype=torch.float32).reshape(1, 1, ACT_DIM)
            with torch.no_grad():
                pred, h_next = model.predict_from_hidden(obs_t, act_t, h)
            nxt, reward, done, truncated, _ = env.step(action)
            target = torch.tensor(np.concatenate([partial(nxt) - obs, [float(reward)]]), dtype=torch.float32)
            target_rmses.append(float((pred[0, 0] - target).square().mean().sqrt()))
            total += float(reward)
            n += 1
            obs = partial(nxt)
            target_obs = nxt
            h = h_next
            if done or truncated:
                break
        env.close()
        returns.append(total)
        lengths.append(n)
    return {
        "condition": condition, "seed": seed, "episodes": episodes,
        "mean_return": float(np.mean(returns)), "returns": returns,
        "mean_length": float(np.mean(lengths)), "lengths": lengths,
        "target_one_step_rmse": float(np.mean(target_rmses)),
    }


def main(args):
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sequences = collect(args.train_seed, args.train_episodes, args.train_steps)
    model = RecurrentWorldModel(args.hidden)
    losses = train_model(model, sequences, args.epochs, args.train_seed)
    digest = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    torch.save(model.state_dict(), out / "model.pt")
    results = []
    for condition in ["clean", "carryover", "reset_defense"]:
        for seed in args.eval_seeds:
            row = evaluate(model, seed, condition, args.eval_episodes, args.eval_steps, args.source_steps,
                           args.cem_horizon, args.cem_population, args.cem_iterations, args.cem_elite)
            results.append(row)
            print(json.dumps(row), flush=True)
    by = {c: [r for r in results if r["condition"] == c] for c in ["clean", "carryover", "reset_defense"]}
    mean = {c: float(np.mean([r["mean_return"] for r in rows])) for c, rows in by.items()}
    summary = {
        "status": "completed_recurrent_context_carryover_gate",
        "candidate": "H_recurrent_belief_carryover_at_episode_boundary",
        "operation": "Legal source prefix initializes a recurrent belief that is retained across a target episode boundary; no observation, reward, label, weight, or planner edit.",
        "carrier": "GRU dynamics model trained on real MBRL-Lib CartPole transitions with velocity hidden from the model; native CartPole closed loop with CEM.",
        "settings": vars(args),
        "training": {"sequences": len(sequences), "final_loss": losses[-1], "initial_loss": losses[0], "epochs": args.epochs, "model_sha256": digest},
        "results": results,
        "mean_returns": mean,
        "relative_carryover_drop_vs_clean": float((mean["clean"] - mean["carryover"]) / max(abs(mean["clean"]), 1e-8)),
        "relative_reset_defense_recovery": float((mean["reset_defense"] - mean["carryover"]) / max(abs(mean["clean"]), 1e-8)),
        "native_environment_steps": int(sum(sum(r["lengths"]) for r in results)),
        "api_requests": 0, "gpu_work": False,
        "claims_excluded": ["cross-task generality", "novelty", "attack success beyond this carrier", "ICLR-scale evidence"],
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "training"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--train-seed", type=int, default=41)
    p.add_argument("--train-episodes", type=int, default=140)
    p.add_argument("--train-steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[501, 502])
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=60)
    p.add_argument("--source-steps", type=int, default=20)
    p.add_argument("--cem-horizon", type=int, default=8)
    p.add_argument("--cem-population", type=int, default=64)
    p.add_argument("--cem-iterations", type=int, default=2)
    p.add_argument("--cem-elite", type=int, default=8)
    main(p.parse_args())
