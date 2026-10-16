"""Short native MBRL-Lib gate for terminal-ownership routing."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env import reward_fns, termination_fns
from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from pilot_mbrl_coupling import make_model, model_hash
from wm_security.terminal_ownership import TerminalOwnershipCoupler


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@torch.no_grad()
def evaluate_particles(model_env, actions, initial, particles):
    population, horizon, _ = actions.shape
    state = model_env.reset(
        np.tile(initial, (population * particles, 1)).astype(np.float32),
        return_as_np=False,
    )
    total = torch.zeros(population * particles, 1)
    done = torch.zeros(population * particles, 1, dtype=torch.bool)
    for t in range(horizon):
        action = torch.repeat_interleave(actions[:, t], particles, dim=0)
        _, reward, terminal, state = model_env.step(action, state, sample=True)
        reward[done] = 0
        done |= terminal
        total += reward
    return total.reshape(population, particles).mean(1)


def run(condition, args, model):
    torch.manual_seed(args.seed)
    env = CartPoleEnv()
    obs, _ = env.reset(args.seed)
    model_env = ModelEnv(
        env,
        model,
        termination_fns.cartpole,
        reward_fns.cartpole,
        generator=torch.Generator().manual_seed(args.seed + 10000),
    )
    coupler = None
    if condition != "native":
        coupler = TerminalOwnershipCoupler(
            termination_fns.cartpole, args.particles, condition, args.seed + 20000
        )
        coupler.install(model_env)
    cfg = OmegaConf.create(
        dict(
            _target_="mbrl.planning.CEMOptimizer",
            num_iterations=args.iterations,
            elite_ratio=args.elite_ratio,
            population_size=args.population,
            alpha=0.1,
            device="cpu",
            return_mean_elites=False,
        )
    )
    agent = TrajectoryOptimizerAgent(
        cfg, [-1.0], [1.0], planning_horizon=args.horizon, replan_freq=1
    )
    agent.set_trajectory_eval_fn(
        lambda initial, actions: evaluate_particles(
            model_env, actions, initial, args.particles
        )
    )
    total = 0.0
    trajectory = []
    try:
        for step in range(args.max_steps):
            action = agent.act(obs)
            nxt, reward, done, truncated, _ = env.step(action)
            trajectory.append(
                dict(
                    step=step,
                    obs=obs.tolist(),
                    action=action.tolist(),
                    next_obs=nxt.tolist(),
                    reward=float(reward),
                    terminated=bool(done),
                )
            )
            total += float(reward)
            obs = nxt
            if done or truncated:
                break
    finally:
        if coupler:
            coupler.restore()
        env.close()
    return dict(
        condition=condition,
        seed=args.seed,
        return_=total,
        steps=len(trajectory),
        survived_cap=len(trajectory) == args.max_steps,
        model_hash_unchanged=True,
        termination_calls=coupler.calls if coupler else 0,
        routed_rows=coupler.routed_rows if coupler else 0,
        native_terminal_rows=coupler.native_terminal_rows if coupler else 0,
    )


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/pilot_mbrl_terminal_ownership.py",
        "src/wm_security/terminal_ownership.py",
        "scripts/pilot_mbrl_coupling.py",
    ]
    model = make_model()
    model.load(Path(args.model))
    model.eval()
    digest = model_hash(model)
    rows = [run(condition, args, model) for condition in ["native", "reverse", "rotate", "random"]]
    assert model_hash(model) == digest
    result = dict(
        status="completed",
        model_hash=digest,
        settings=vars(args),
        rows=rows,
        api_requests=0,
        training_updates=0,
        gpu_work=False,
        interpretation=(
            "Native state and reward predictions are unchanged; terminal bits are routed "
            "between candidate groups while preserving their batch multiset. Promote only "
            "if structured routing harms both seeds against random routing and a native gate."
        ),
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(out / "summary.json", result)
    save(
        out / "plan.json",
        dict(
            operation="Within each CEM step, route native terminal vectors between action-candidate groups; state, reward, weights, and the batch terminal multiset stay unchanged.",
            scope="Frozen native MBRL-Lib five-member GaussianMLP, CartPole, short CPU gate.",
            controls="Native, action-rank reversal, cyclic routing, and random group routing.",
            promotion_gate="Require repeatable native task harm from structured routing against random and independent seeds before any scale-up.",
            hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            api_requests=0,
            gpu_work=False,
        ),
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--particles", type=int, default=5)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--elite-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=810)
    main(parser.parse_args())
