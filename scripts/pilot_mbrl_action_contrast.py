"""Short native MBRL-Lib action-contrast routing screen."""
from __future__ import annotations

import argparse
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
from wm_security.action_contrast import ActionContrastCoupler


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


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
    seed = args.seed
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
            alpha=0.1,
            device="cpu",
            return_mean_elites=False,
        )
    )
    agent = TrajectoryOptimizerAgent(
        cfg, [-1.0], [1.0], planning_horizon=args.horizon, replan_freq=1
    )
    coupler = None
    if condition != "native":
        coupler = ActionContrastCoupler(model, args.particles, condition)
        coupler.install()

    def objective(initial, actions):
        return evaluate_particles(model_env, actions, initial, args.particles)

    agent.set_trajectory_eval_fn(objective)
    total = 0.0
    steps = 0
    try:
        for _ in range(args.max_steps):
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if done or truncated:
                break
    finally:
        if coupler:
            coupler.restore()
        env.close()
    return {
        "condition": condition,
        "seed": seed,
        "return_": total,
        "steps": steps,
        "survived_cap": steps == args.max_steps,
        "model_hash_unchanged": True,
        "model_calls": coupler.calls if coupler else 0,
        "routed_rows": coupler.routed_rows if coupler else 0,
    }


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = make_model()
    model.load(Path(args.model))
    model.eval()
    digest = model_hash(model)
    rows = [run(condition, args, model) for condition in ["native", "reverse", "rotate", "random"]]
    assert model_hash(model) == digest
    result = {
        "status": "completed",
        "model_hash": digest,
        "settings": vars(args),
        "rows": rows,
        "api_requests": 0,
        "training_updates": 0,
        "gpu_work": False,
        "interpretation": "One-step output multisets are routed across action candidates. Reverse and rotate do not harm this short CEM gate; random routing does, so the candidate does not establish action-contrast-specific security.",
    }
    save(out / "summary.json", result)
    save(
        out / "plan.json",
        {
            "operation": "Within each CEM candidate's particle block, permute native next-state outputs across action rows while leaving the output multiset unchanged.",
            "scope": "Frozen native MBRL-Lib GaussianMLP, CartPole, short CPU gate.",
            "controls": "Native, reverse action-rank routing, cyclic routing, and random routing.",
            "promotion_gate": "Only promote if structured routing harms both seeds and matched random/ordinary action perturbation controls do not.",
        },
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--particles", type=int, default=5)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--elite-ratio", type=float, default=0.2)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=810)
    main(p.parse_args())
