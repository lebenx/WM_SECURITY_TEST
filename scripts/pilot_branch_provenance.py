"""Short recurrent-world-model gate for provenance-confounded branch splicing.

The attacker replaces a victim observation with a cue that is authentic in a
different episode.  Source authenticity does NOT imply truth for the victim.
An exact Bayesian planner using the original observation law is included to
test whether any loss needs a learned world model at all.

This is a mechanism screen.  The tiny POMDP and enumerating planner are not a
core-task result; promotion requires native recurrent world-model benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BranchWorld:
    OBS_DIM = 2
    ACTIONS = 4  # sense, wait, choose branch 0, choose branch 1

    def __init__(self, mode: int):
        self.mode = int(mode)

    def episode(self):
        obs = [np.array([0.0, 0.0], dtype=np.float32)]
        actions = [0, 1]
        # The cue is a genuine observation in this branch and disappears after
        # the wait transition.  The final choice is scored by the hidden mode.
        obs.append(np.array([1.0, float(self.mode)], dtype=np.float32))
        obs.append(np.array([0.0, 0.0], dtype=np.float32))
        return obs, actions

    def reward(self, action: int) -> float:
        return float(action - 2 == self.mode)


class RecurrentWorldModel(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(2 + BranchWorld.ACTIONS, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, features):
        h, _ = self.gru(features)
        return self.head(h)


class OneStepWorldModel(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + BranchWorld.ACTIONS, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, features):
        return self.net(features)


def features(observations, actions):
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    act = torch.as_tensor(np.asarray(actions), dtype=torch.long)
    return torch.cat([obs, torch.nn.functional.one_hot(act, BranchWorld.ACTIONS).float()], -1)


def make_dataset(count: int, seed: int):
    rng = np.random.default_rng(seed)
    observations, actions, targets = [], [], []
    rows = []
    for _ in range(count):
        mode = int(rng.integers(0, 2))
        world = BranchWorld(mode)
        obs, fixed_actions = world.episode()
        final_action = int(rng.integers(2, 4))
        full_actions = fixed_actions + [final_action]
        next_obs = obs[1:] + [np.array([0.0, 0.0], dtype=np.float32)]
        rewards = [0.0, 0.0, world.reward(final_action)]
        observations.append(obs)
        actions.append(full_actions)
        targets.append(np.concatenate([np.asarray(next_obs), np.asarray(rewards)[:, None]], -1))
        for t in range(3):
            rows.append(
                (
                    tuple(np.asarray(obs[t]).round(6)),
                    int(full_actions[t]),
                    tuple(np.asarray(next_obs[t]).round(6)),
                    float(rewards[t]),
                )
            )
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
        rows,
    )


def train_model(model, x, y, epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def sequence_features(obs_batch, action_batch):
    obs = torch.as_tensor(obs_batch, dtype=torch.float32)
    act = torch.as_tensor(action_batch, dtype=torch.long)
    return torch.cat([obs, torch.nn.functional.one_hot(act, BranchWorld.ACTIONS).float()], -1)


def fit(seed: int, count: int, epochs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, actions, targets, rows = make_dataset(count, seed + 100)
    split = int(0.8 * count)
    train_x = sequence_features(obs[:split], actions[:split])
    train_y = torch.as_tensor(targets[:split], dtype=torch.float32)
    valid_x = sequence_features(obs[split:], actions[split:])
    valid_y = torch.as_tensor(targets[split:], dtype=torch.float32)
    recurrent = RecurrentWorldModel()
    one_step = OneStepWorldModel()
    rec_loss = train_model(recurrent, train_x, train_y, epochs, 1e-2)
    # One-step model sees exactly the same transition rows, independently of
    # sequence order.  It is a matched non-recurrent audit baseline.
    flat_x = sequence_features(obs[:split, :3].reshape(-1, 2), actions[:split].reshape(-1))
    flat_y = train_y.reshape(-1, 3)
    opt = torch.optim.Adam(one_step.parameters(), lr=1e-2)
    for _ in range(epochs):
        pred = one_step(flat_x)
        loss = ((pred - flat_y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec_valid = float(((recurrent(valid_x) - valid_y) ** 2).mean())
        rec_reward_mse = float(((recurrent(valid_x)[:, 2, 2] - valid_y[:, 2, 2]) ** 2).mean())
        one_valid = float(((one_step(sequence_features(obs[split:, :3].reshape(-1, 2), actions[split:].reshape(-1))) - valid_y.reshape(-1, 3)) ** 2).mean())
    return recurrent, one_step, rows, dict(train_recurrent_mse=rec_loss, valid_recurrent_mse=rec_valid, valid_recurrent_final_reward_mse=rec_reward_mse, valid_one_step_mse=one_valid)


def planner_action(model, observations, final_action):
    actions = [0, 1, final_action]
    with torch.no_grad():
        pred = model(sequence_features(np.asarray(observations, dtype=np.float32)[None], np.asarray(actions, dtype=np.int64)[None]))
    return float(pred[0, 2, 2])


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def local_tuple(obs, action, next_obs, reward):
    return (tuple(float(v) for v in obs), int(action), tuple(float(v) for v in next_obs), float(reward))


def exact_open_loop_audit():
    """Exhaust all 4 fixed mode/action pairs; no adaptive action selection."""
    tables = {}
    histories = {}
    for condition in ["clean", "opposite_valid_splice"]:
        table = Counter()
        full = Counter()
        for mode in [0, 1]:
            for final in [2, 3]:
                world = BranchWorld(mode)
                obs, _ = world.episode()
                if condition != "clean":
                    obs[1] = BranchWorld(1 - mode).episode()[0][1]
                tuples = []
                for t, action in enumerate([0, 1, final]):
                    nxt = obs[t + 1] if t < 2 else [0., 0.]
                    row = local_tuple(obs[t], action, nxt, world.reward(final) if t == 2 else 0.)
                    table[(t, row)] += 1
                    tuples.append(row)
                full[tuple(tuples)] += 1
        tables[condition] = table
        histories[condition] = full
    return {
        "fixed_action_audit_local_tuple_multiset_equal": tables["clean"] == tables["opposite_valid_splice"],
        "fixed_action_audit_full_history_multiset_equal": histories["clean"] == histories["opposite_valid_splice"],
        "scope": "Exact enumeration of balanced hidden modes and independently fixed final actions; NOT preservation of conditional history laws or closed-loop marginals.",
    }


def evaluate(recurrent, one_step, clean_support, seed: int, episodes: int):
    rng = np.random.default_rng(seed + 9000)
    rows = []
    # Fix exactly the same replacement schedule for every decision rule.
    random_cues = rng.integers(0, 2, size=episodes)
    for condition in ["clean", "opposite_valid_splice", "random_valid_splice"]:
      for planner in ["recurrent", "one_step", "exact_bayes", "attack_aware_bayes"]:
        successes, changed_cues = 0, 0
        tuples, decisions = [], []
        for i in range(episodes):
            mode = i % 2
            world = BranchWorld(mode)
            obs, _ = world.episode()
            supplied = [np.array(v, copy=True) for v in obs]
            if condition == "opposite_valid_splice":
                supplied[1] = BranchWorld(1 - mode).episode()[0][1].copy()
            elif condition == "random_valid_splice":
                supplied[1] = BranchWorld(int(random_cues[i])).episode()[0][1].copy()
            changed_cues += int(not np.array_equal(supplied[1], obs[1]))
            if planner == "one_step":
                scores = [planner_action(one_step, supplied, a) for a in [2, 3]]
                choice = int(np.argmax(scores))
            elif planner == "recurrent":
                scores = [planner_action(recurrent, supplied, a) for a in [2, 3]]
                choice = int(np.argmax(scores))
            else:
                # Exact posterior under clean observation law: P(mode=cue)=1.
                # Attack-aware control is granted the replacement law, NOT mode.
                cue = int(supplied[1][1])
                choice = cue
                if planner == "attack_aware_bayes":
                    if condition == "opposite_valid_splice":
                        choice = 1 - cue
                    elif condition == "random_valid_splice":
                        choice = 0  # uniform posterior, deterministic tie break
                scores = None
            successes += int(choice == mode)
            decisions.append(dict(mode=mode, cue=int(supplied[1][1]), choice=choice, scores=scores))
            for t in range(3):
                next_obs = supplied[t + 1] if t < 2 else np.array([0.0, 0.0], dtype=np.float32)
                reward = world.reward(choice + 2) if t == 2 else 0.0
                tuples.append(local_tuple(supplied[t], [0, 1, choice + 2][t], next_obs, reward))
        rows.append({
            "condition": condition,
            "planner": planner,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "all_local_tuples_present_in_clean_support": all(row in clean_support for row in tuples),
            "victim_observations_changed": changed_cues,
            "one_step_rows": len(tuples),
            "decisions": decisions,
        })
    return rows


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if args.eval_episodes % 2:
        raise ValueError("Use an even evaluation count to balance hidden modes.")
    started = time.monotonic()
    plan = {
        "operation": "Replace the victim cue using another episode's valid observation. This edits the victim's observed history, despite source authenticity.",
        "scope": "Synthetic mechanism reduction, four distinct mode/action cases; repeated evaluations do not represent independent tasks or native benchmarks.",
        "seeds": args.seeds,
        "controls": ["recurrent", "one_step", "exact_bayes", "attack_aware_bayes"],
        "decision_rule": "If exact Bayes with unmodified world dynamics exhibits the same clean-to-attack loss as the recurrent model, reject learned-world-model-specific causation for this construction.",
        "preservation_scope": "Audit balanced fixed-action local tuples separately; do not infer preservation conditional on history or under closed-loop actions.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "api_requests": 0, "gpu_work": False,
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    all_runs = []
    for seed in args.seeds:
        recurrent, one_step, clean_rows, fit_stats = fit(seed, args.train_episodes, args.epochs)
        hashes = {"recurrent": model_hash(recurrent), "one_step": model_hash(one_step)}
        rows = evaluate(recurrent, one_step, set(clean_rows), seed, args.eval_episodes)
        assert hashes == {"recurrent": model_hash(recurrent), "one_step": model_hash(one_step)}
        assert all(r["all_local_tuples_present_in_clean_support"] for r in rows)
        torch.save({"recurrent": recurrent.state_dict(), "one_step": one_step.state_dict()}, out / f"models_{seed}.pt")
        (out / f"decisions_{seed}.json").write_text(json.dumps(rows, indent=2) + "\n")
        brief = [{k: v for k, v in r.items() if k != "decisions"} for r in rows]
        all_runs.append({"seed": seed, "fit": fit_stats, "rows": brief, "model_hashes": hashes, "evaluation_weights_unchanged": True})
        print(json.dumps(all_runs[-1]), flush=True)
    result = {
        "status": "completed",
        "candidate": "H_provenance_confounded_branch_splice",
        "runs": all_runs,
        "settings": vars(args),
        "api_requests": 0,
        "gpu_work": False,
        "training_updates": len(args.seeds) * args.epochs * 2,
        "audit": exact_open_loop_audit(),
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Source-valid cue replacement is still victim observation corruption. Exact-Bayes and attack-aware controls separate this from learned world-model failure. The toy has only four distinct mode/action cases, no native or broad-task evidence.",
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23])
    p.add_argument("--train-episodes", type=int, default=2500)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=160)
    main(p.parse_args())
