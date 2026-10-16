"""Short native MBRL-Lib gate for trajectory-order memory.

This uses the upstream CEM and ModelEnv with a frozen five-member GaussianMLP.
The intervention is inference-only and leaves physical dynamics, rewards, and
model parameters untouched.  It is a candidate screen, not a claim of broad
task harm or novelty.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.env import reward_fns, termination_fns
from mbrl.models import ModelEnv
from mbrl.planning import TrajectoryOptimizerAgent
from pilot_mbrl_coupling import make_model, model_hash
from wm_security.trajectory_order import TrajectoryOrderCoupler


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def clone_state(state):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}


@torch.no_grad()
def evaluate_particles(model_env, actions, initial_state, particles):
    population, horizon, _ = actions.shape
    init = np.tile(initial_state, (particles * population, 1)).astype(np.float32)
    state = model_env.reset(init, return_as_np=False)
    total = torch.zeros(particles * population, 1)
    terminated = torch.zeros(particles * population, 1, dtype=torch.bool)
    for t in range(horizon):
        action = torch.repeat_interleave(actions[:, t], particles, dim=0)
        _, rewards, dones, state = model_env.step(action, state, sample=True)
        rewards[terminated] = 0
        terminated |= dones
        total += rewards
    return total.reshape(population, particles)


def one_step_parity(model_env, coupler) -> float:
    """Check that a fresh one-step deterministic call is bitwise unchanged."""
    initial = np.zeros((5, 4), dtype=np.float32)
    state = model_env.reset(initial, return_as_np=False)
    action = torch.zeros(5, 1)
    original_state = clone_state(state)
    expected = coupler.original_sample(
        action, original_state, deterministic=True, rng=None
    )[0]
    actual = coupler.sample(action, clone_state(state), deterministic=True, rng=None)[0]
    return float((expected - actual).abs().max().detach().cpu())


def evaluate(args, out: Path) -> None:
    model = make_model()
    model.load(Path(args.model))
    model.eval()
    before = model_hash(model)
    rows = []
    started = time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        env = CartPoleEnv()
        obs, _ = env.reset(seed)
        model_env = ModelEnv(
            env,
            model,
            termination_fns.cartpole,
            reward_fns.cartpole,
            generator=torch.Generator().manual_seed(seed + 10000),
        )
        cfg = OmegaConf.create(
            dict(
                _target_="mbrl.planning.CEMOptimizer",
                num_iterations=args.iterations,
                elite_ratio=args.elite_ratio,
                population_size=args.population,
                alpha=args.alpha,
                device="cpu",
                return_mean_elites=False,
            )
        )
        agent = TrajectoryOptimizerAgent(
            cfg, [-1.0], [1.0], planning_horizon=args.horizon, replan_freq=1
        )
        coupler = TrajectoryOrderCoupler(
            model,
            strength=args.strength,
            target_dim=args.target_dim,
            signed=args.condition == "order",
        )
        parity = one_step_parity(model_env, coupler)
        if args.condition != "native":
            coupler.install()

        def objective(initial, action_sequences):
            particles = evaluate_particles(
                model_env, action_sequences, initial, args.particles
            )
            return particles.mean(dim=1)

        agent.set_trajectory_eval_fn(objective)
        total = 0.0
        trajectory = []
        episode_start = time.monotonic()
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
            if args.condition != "native":
                coupler.restore()
            env.close()
        assert model_hash(model) == before
        row = dict(
            condition=args.condition,
            seed=seed,
            strength=args.strength,
            return_=total,
            steps=len(trajectory),
            survived_cap=len(trajectory) == args.max_steps
            and not trajectory[-1]["terminated"],
            one_step_max_abs_error=parity,
            model_hash_unchanged=True,
            model_calls=coupler.calls,
            modified_batch_rows=coupler.modified_calls,
            mean_abs_order_phase=coupler.phase_abs_sum / max(1, coupler.modified_calls),
            max_abs_order_phase=coupler.max_phase,
            seconds=time.monotonic() - episode_start,
        )
        rows.append(row)
        save(out / f"episode_{seed}.json", dict(summary=row, trajectory=trajectory))
        print(json.dumps(row), flush=True)
    result = dict(
        status="completed",
        condition=args.condition,
        rows=rows,
        model_hash=before,
        mean_return=float(np.mean([r["return_"] for r in rows])),
        elapsed_seconds=time.monotonic() - started,
        environment_steps=sum(r["steps"] for r in rows),
        api_requests=0,
        training_updates=0,
        gpu_work=False,
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def main(args) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    files = [
        "scripts/pilot_mbrl_trajectory_order.py",
        "src/wm_security/trajectory_order.py",
        "scripts/pilot_mbrl_coupling.py",
    ]
    save(
        out / "plan.json",
        dict(
            arguments=vars(args),
            hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
            scope="Frozen native MBRL-Lib five-member GaussianMLP and upstream CEM; CartPole short gate.",
            operation="Retain previous imagined action in model state and add a signed action-order term after the first imagined step; physics, reward, and weights remain unchanged.",
            preservation="Fresh one-step deterministic calls are unchanged exactly; this does not claim preservation of conditional multi-step transition laws.",
            promotion_gate="Only a bounded mechanism screen. Do not scale unless native control is strong, one-step parity passes, and order attack produces repeatable task-specific loss against an action-blind and matched perturbation control.",
            api_requests=0,
            gpu_work=False,
        ),
    )
    evaluate(args, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--condition", choices=["native", "order", "unsigned"], required=True
    )
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--target-dim", type=int, default=2)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite-ratio", type=float, default=0.1)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[810, 811])
    main(parser.parse_args())
