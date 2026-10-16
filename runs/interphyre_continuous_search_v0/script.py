"""Native continuous-search comparators; no novelty or HExA reproduction claim.

Uses only legal initial placements and observed trajectories. Every placement,
including initialization and invalid proposals, consumes the same query budget.
The native terminal predicate is unmodified; progress scores only guide search.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time

import cma
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from interphyre import InterphyreEnv


LOW = np.array([-5., -5., .1])
SPAN = np.array([10., 10., 1.4])


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(a):
    return np.clip((np.asarray(a) - LOW) / SPAN, 0, 1)


def object_prior(scene, rng, count=1):
    objects = [o for o in scene["objects"].values()
               if o["dynamic"] and o["color"] != "red"]
    result = []
    for _ in range(count):
        r = rng.uniform(.15, 1.5)
        if rng.random() < .8:
            obj = objects[int(rng.integers(len(objects)))]
            x = obj["x"] + rng.normal(0, max(.5, r))
            top = obj["y"] + obj["size"].get("radius", 0)
            y = rng.uniform(min(top + r + .05, 5 - r), 5 - r)
        else:
            x, y = rng.uniform(-5 + r, 5 - r, 2)
        result.append(normalize([np.clip(x, -5 + r, 5 - r), y, r]))
    return np.array(result)


def evaluate(env, u):
    env.reset()
    action = (LOW + np.clip(u, 0, 1) * SPAN).tolist()
    valid = env.validate_action([action])
    if valid["invalid"]:
        return dict(action=action, normalized=np.asarray(u).tolist(), invalid=True,
                    success=False, score=-1., physics_steps=0, min_gap=None,
                    max_contact_duration=0., first_action_contact=None)
    initial = env.describe_scene()["objects"]
    radius_sum = sum(initial[name]["size"]["radius"]
                     for name in ("green_ball", "blue_ball"))
    gp = np.asarray(env.get_object_position("green_ball"))
    bp = np.asarray(env.get_object_position("blue_ball"))
    initial_gap = max(float(np.linalg.norm(gp - bp)) - radius_sum, .01)
    min_gap, max_contact = initial_gap, 0.
    best_error = (gp - bp).tolist()
    env.place_action([action])
    success = False
    for _ in range(env.max_steps):
        env.step_physics()
        gp = env.get_object_position("green_ball")
        bp = env.get_object_position("blue_ball")
        error = (gp[0] - bp[0], gp[1] - bp[1])
        gap = max(float(np.hypot(*error)) - radius_sum, 0.)
        if gap < min_gap:
            min_gap, best_error = gap, list(error)
        max_contact = max(max_contact, env.engine.get_contact_duration("green_ball", "blue_ball"))
        if env.success:
            success = True
            break
    first = None
    for event in env.get_contact_log():
        if event.get("event") == "begin" and "red_ball" in event.get("objects", []):
            first = next((x for x in event["objects"] if x != "red_ball"), None)
            break
    score = 2. if success else (1 - min_gap / initial_gap + min(max_contact / 3., 1.))
    return dict(action=action, normalized=np.asarray(u).tolist(), invalid=False,
                success=success, score=float(score), physics_steps=env.step_count,
                min_gap=min_gap, max_contact_duration=max_contact,
                closest_error=best_error, first_action_contact=first)


def bo_proposal(history, scene, rng):
    x = np.array([r["normalized"] for r in history])
    y = np.array([r["score"] for r in history])
    best = x[int(np.argmax(y))]
    candidates = np.vstack([object_prior(scene, rng, 128),
                            np.clip(best + rng.normal(0, .1, (128, 3)), 0, 1)])
    length = np.array([.15, .2, .25])
    distance = np.sum(((x[:, None] - x[None]) / length) ** 2, axis=-1)
    kernel = np.exp(-.5 * distance) + np.eye(len(x)) * .01
    cross = np.exp(-.5 * np.sum(((candidates[:, None] - x[None]) / length) ** 2, axis=-1))
    factor = cho_factor(kernel)
    mean = y.mean() + cross @ cho_solve(factor, y - y.mean())
    variance = np.maximum(1 - np.sum(cross * cho_solve(factor, cross.T).T, axis=1), 1e-8)
    score = mean + .5 * np.sqrt(variance)
    nearest = np.min(np.linalg.norm(candidates[:, None] - x[None], axis=-1), axis=1)
    score[nearest < 1e-6] = -np.inf
    return candidates[int(np.argmax(score))]


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(levels=args.levels, seeds=list(range(args.start_seed, args.start_seed + args.seeds)),
                arms=["object_prior", "cma_es", "rbf_bo"], queries=args.queries,
                initialization_queries=args.initialization, native_steps=1000,
                score="1 - minimum_surface_gap / initial_surface_gap + maximum_continuous_contact / 3; native success=2; invalid=-1",
                scope="Small native development comparison, not a new method or faithful Causal-PIK/HExA reproduction.",
                acquisition_cost="All initialization, invalid, and subsequent placements count. No hidden candidate simulations.",
                api_requests=0, gpu_work=False)
    save(out / "plan.json", plan)
    started = time.monotonic()
    results, counts = [], dict(placement_queries=0, valid_rollouts=0, invalid_queries=0, physics_steps=0)
    with (out / "native_traces.jsonl").open("w") as log:
        for level_idx, level in enumerate(args.levels):
            for seed in plan["seeds"]:
                for arm_idx, arm in enumerate(plan["arms"]):
                    env = InterphyreEnv(level, seed=seed)
                    env.reset()
                    scene = env.describe_scene()
                    # Initial proposals are exactly paired across the arms.
                    base_seed = 10101 + seed + level_idx * 10000
                    initial = object_prior(scene, np.random.default_rng(base_seed), args.initialization)
                    rng = np.random.default_rng(base_seed + 100000)
                    history, optimizer, pending, batch, losses = [], None, [], [], []
                    for step in range(args.queries):
                        if step < args.initialization:
                            u = initial[step]
                        elif arm == "object_prior":
                            u = object_prior(scene, rng)[0]
                        elif arm == "rbf_bo":
                            u = bo_proposal(history, scene, rng)
                        else:
                            if optimizer is None:
                                best = max(history, key=lambda r: r["score"])["normalized"]
                                optimizer = cma.CMAEvolutionStrategy(best, .15,
                                    {"bounds": [0., 1.], "popsize": 4, "seed": base_seed,
                                     "verbose": -9, "verb_log": 0})
                            if not pending:
                                pending = optimizer.ask()
                                batch, losses = [], []
                            u = np.asarray(pending.pop(0))
                        record = evaluate(env, u)
                        record.update(level=level, seed=seed, arm=arm, query=step + 1)
                        history.append(record)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        counts["placement_queries"] += 1
                        counts["valid_rollouts"] += int(not record["invalid"])
                        counts["invalid_queries"] += int(record["invalid"])
                        counts["physics_steps"] += record["physics_steps"]
                        if arm == "cma_es" and optimizer is not None:
                            batch.append(u)
                            losses.append(-record["score"])
                            if not pending:
                                optimizer.tell(batch, losses)
                        if record["success"]:
                            break
                    env.close()
                    result = dict(level=level, seed=seed, arm=arm,
                                  success=bool(history[-1]["success"]), queries=len(history),
                                  best_score=max(r["score"] for r in history),
                                  valid_rollouts=sum(not r["invalid"] for r in history))
                    results.append(result)
                    print(json.dumps(result), flush=True)
    save(out / "task_results.json", results)
    aggregate = {}
    for arm in plan["arms"]:
        selected = [r for r in results if r["arm"] == arm]
        aggregate[arm] = dict(successes=sum(r["success"] for r in selected), tasks=len(selected),
                              queries=sum(r["queries"] for r in selected))
    summary = dict(status="completed", run=str(out), counts=counts, results=aggregate,
                   elapsed_s=time.monotonic() - started, api_requests=0, gpu_work=False,
                   novel_method=False, scope=plan["scope"],
                   finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels", nargs="+", default=["catapult", "pass_the_parcel"])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=300)
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--initialization", type=int, default=5)
    run(parser.parse_args())
