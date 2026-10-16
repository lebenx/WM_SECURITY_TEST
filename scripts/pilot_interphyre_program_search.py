"""Learn bounded branching experiment programs from native source outcomes.

This is a mechanism feasibility screen, not a new-method or HExA reproduction.
All methods receive the same source table and action-expression library. Target
outcomes are obtained only for actions requested by each frozen policy. Native
solutions and the seed registry are not exposed to the learner.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np
from interphyre import InterphyreEnv


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def halton(i, base):
    out, scale = 0.0, 1.0
    while i:
        scale /= base
        out += scale * (i % base)
        i //= base
    return out


def library():
    # Generic placement expressions, shared unchanged with all comparators.
    out = [{"op": "absolute", "xyz": [10 * halton(i, 2) - 5,
            10 * halton(i, 3) - 5, .1 + 1.4 * halton(i, 5)]}
           for i in range(1, 33)]
    for rank in range(4):
        for offset in (-.8, .8):
            for radius in (.6, 1.2):
                out.append({"op": "above_dynamic", "rank": rank,
                            "offset": offset, "radius": radius})
    return out


def action(expr, scene):
    if expr["op"] == "absolute":
        return expr["xyz"]
    objects = [(name, obj) for name, obj in scene["objects"].items()
               if obj["dynamic"] and obj["color"] != "red"]
    objects.sort(key=lambda x: (x[1]["x"], x[1]["y"], x[0]))
    obj = objects[min(expr["rank"], len(objects) - 1)][1]
    r = expr["radius"]
    return [float(np.clip(obj["x"] + expr["offset"], -5 + r, 5 - r)),
            5 - r - .05, r]


def geometry(scene):
    # Object names never enter a predictor; public positions/shapes do.
    objects = sorted(scene["objects"].values(),
                     key=lambda x: (x["color"], x["type"], x["x"], x["y"]))
    return np.asarray([[o["x"] / 10, o["y"] / 10,
                        float(o["dynamic"]), *[o["size"].get(k, 0) / 10
                         for k in ("radius", "length", "height")]]
                       for o in objects], dtype=float).ravel()


def observation(scene, info):
    # Both selected levels explicitly ask for sustained green/blue contact.
    g, b = scene["objects"]["green_ball"], scene["objects"]["blue_ball"]
    return np.asarray([float(info.get("invalid_action", False)),
                       g["x"] / 10, g["y"] / 10,
                       b["x"] / 10, b["y"] / 10,
                       (g["x"] - b["x"]) / 10,
                       (g["y"] - b["y"]) / 10,
                       np.hypot(g["x"] - b["x"], g["y"] - b["y"]) / 10])


def execute(env, expr, log, phase, level, seed, arm, index):
    env.reset()
    initial = env.describe_scene()
    placement = action(expr, initial)
    _, reward, _, _, info = env.step([placement])
    obs = observation(env.describe_scene(), info)
    record = dict(phase=phase, level=level, seed=seed, arm=arm,
                  action_index=index, placement=placement, reward=float(reward),
                  success=bool(info["success"]), invalid=bool(info.get("invalid_action", False)),
                  physics_steps=info["step_count"], observation=obs.tolist())
    log.write(json.dumps(record) + "\n")
    log.flush()
    return record


def greedy_sequence(y, rows, excluded, depth):
    rows = list(rows)
    sequence, count = [], 0
    for _ in range(depth):
        choices = [a for a in range(y.shape[1]) if a not in excluded and a not in sequence]
        if not choices or not rows:
            break
        a = max(choices, key=lambda x: (int(y[rows, x].sum()), -x))
        sequence.append(a)
        count += int(y[rows, a].sum())
        rows = [i for i in rows if not y[i, a]]
    return count, {"op": "sequence", "actions": sequence}


def synthesize(y, obs, rows, excluded, depth):
    """Induce action choice, branch feature/threshold, and subtree structure.

    A beam of one branching split per node is scored by greedy continuation
    coverage. A node is kept only when it improves source solved-task coverage
    relative to an equally long unconditioned sequence.
    """
    baseline_count, baseline = greedy_sequence(y, rows, excluded, depth)
    if depth < 2 or len(rows) < 3:
        return baseline_count, baseline
    best = None
    for a in range(y.shape[1]):
        if a in excluded:
            continue
        remaining = [i for i in rows if not y[i, a]]
        immediate = len(rows) - len(remaining)
        if len(remaining) < 2:
            continue
        for f in range(obs.shape[2]):
            values = sorted(set(obs[remaining, a, f]))
            for lo, hi in zip(values[:-1], values[1:]):
                threshold = (lo + hi) / 2
                left = [i for i in remaining if obs[i, a, f] <= threshold]
                right = [i for i in remaining if obs[i, a, f] > threshold]
                n1, _ = greedy_sequence(y, left, excluded | {a}, depth - 1)
                n2, _ = greedy_sequence(y, right, excluded | {a}, depth - 1)
                score = immediate + n1 + n2
                if score > baseline_count and (best is None or score > best[0]):
                    best = (score, a, f, float(threshold), left, right)
    if best is None:
        return baseline_count, baseline
    _, a, f, threshold, left, right = best
    n1, p1 = synthesize(y, obs, left, excluded | {a}, depth - 1)
    n2, p2 = synthesize(y, obs, right, excluded | {a}, depth - 1)
    immediate = sum(bool(y[i, a]) for i in rows)
    return immediate + n1 + n2, {"op": "experiment_then_branch", "action": a,
            "feature": f, "threshold": threshold, "left": p1, "right": p2}


def kernel_choice(y, obs, geo, target_geo, history):
    # Same-data ordinary predictive comparator. Scene and outcome similarities
    # update weights on source environments, then maximize predicted success.
    distances = np.mean((geo - target_geo) ** 2, axis=1)
    scene_scale = max(float(np.median(distances)), 1e-4)
    energy = distances / scene_scale
    used = set()
    for a, actual_obs in history:
        used.add(a)
        dist = np.mean((obs[:, a] - actual_obs) ** 2, axis=1)
        energy += dist / max(float(np.median(dist)), 1e-4)
        # A failed query rules against source worlds where it succeeds.
        energy += y[:, a] * 2.0
    weights = np.exp(-(energy - energy.min()))
    weights /= weights.sum()
    scores = weights @ y
    scores[list(used)] = -1
    return int(np.argmax(scores))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    expressions = (json.loads(Path(args.candidate_file).read_text())["action_expressions"]
                   if args.candidate_file else library())
    plan = dict(levels=args.levels, source_seeds=list(range(args.source_seeds)),
                target_seeds=list(range(args.target_start, args.target_start + args.target_seeds)),
                action_expressions=expressions, target_query_budget=args.budget,
                candidate_file=args.candidate_file,
                native_max_steps=1000, api_requests=0, gpu_work=False,
                scope="Within-level unseen-seed development screen; not cross-level transfer or a new-method claim.")
    save(out / "plan.json", plan)
    start = time.monotonic()
    results = []
    with (out / "native_traces.jsonl").open("w") as log:
        for level in args.levels:
            y, obs, geo = [], [], []
            for seed in plan["source_seeds"]:
                env = InterphyreEnv(level, seed=seed)
                env.reset()
                geo.append(geometry(env.describe_scene()))
                rows = [execute(env, expr, log, "source", level, seed, "shared", i)
                        for i, expr in enumerate(expressions)]
                y.append([r["success"] for r in rows])
                obs.append([r["observation"] for r in rows])
                env.close()
                print(json.dumps(dict(phase="source", level=level, seed=seed,
                                     successes=sum(y[-1]), candidates=len(rows))), flush=True)
            y, obs, geo = np.asarray(y, bool), np.asarray(obs), np.asarray(geo)
            count, program = synthesize(y, obs, list(range(len(y))), set(), args.budget)
            fixed_count, fixed = greedy_sequence(y, list(range(len(y))), set(), args.budget)
            save(out / f"{level}_source.json", dict(success=y.tolist(), observations=obs.tolist(),
                 geometry=geo.tolist(), learned_program=program, source_solved=count,
                 fixed_sequence=fixed, fixed_source_solved=fixed_count,
                 frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            for seed in plan["target_seeds"]:
                for arm in ("learned_program", "same_data_kernel_model", "source_greedy_sequence", "library_prefix"):
                    env = InterphyreEnv(level, seed=seed)
                    env.reset()
                    target_geo = geometry(env.describe_scene())
                    history, records, node, seq_pos = [], [], program, 0
                    for step in range(args.budget):
                        if arm == "same_data_kernel_model":
                            a = kernel_choice(y, obs, geo, target_geo, history)
                        elif arm == "source_greedy_sequence":
                            if step >= len(fixed["actions"]):
                                break
                            a = fixed["actions"][step]
                        elif arm == "library_prefix":
                            a = step
                        elif node["op"] == "sequence":
                            if seq_pos >= len(node["actions"]):
                                break
                            a = node["actions"][seq_pos]
                            seq_pos += 1
                        else:
                            a = node["action"]
                        record = execute(env, expressions[a], log, "target", level, seed, arm, a)
                        records.append(record)
                        actual = np.asarray(record["observation"])
                        history.append((a, actual))
                        if record["success"]:
                            break
                        if arm == "learned_program" and node["op"] != "sequence":
                            node = node["left"] if actual[node["feature"]] <= node["threshold"] else node["right"]
                            seq_pos = 0
                    env.close()
                    result = dict(level=level, seed=seed, arm=arm, queries=len(records),
                                  success=any(r["success"] for r in records),
                                  actions=[r["action_index"] for r in records])
                    results.append(result)
                    print(json.dumps(dict(phase="target", **result)), flush=True)
    save(out / "target_results.json", results)
    traces = [json.loads(line) for line in (out / "native_traces.jsonl").read_text().splitlines()]
    totals = {}
    for arm in sorted(set(r["arm"] for r in results)):
        rows = [r for r in results if r["arm"] == arm]
        totals[arm] = dict(successes=sum(r["success"] for r in rows), tasks=len(rows),
                           placement_queries=sum(r["queries"] for r in rows))
    summary = dict(status="completed", run=str(out), elapsed_s=time.monotonic() - start,
                   native_placement_queries=len(traces),
                   valid_physics_rollouts=sum(not r["invalid"] for r in traces),
                   invalid_placement_queries=sum(r["invalid"] for r in traces),
                   physics_steps=sum(r["physics_steps"] for r in traces),
                   source_queries=sum(r["phase"] == "source" for r in traces),
                   target_queries=sum(r["phase"] == "target" for r in traces),
                   results=totals, api_requests=0, gpu_work=False,
                   method_novelty_established=False, strong_baseline_advantage_established=False,
                   scope=plan["scope"],
                   finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels", nargs="+", default=["catapult", "pass_the_parcel"])
    parser.add_argument("--source-seeds", type=int, default=8)
    parser.add_argument("--target-seeds", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--candidate-file")
    parser.add_argument("--target-start", type=int, default=100)
    run(parser.parse_args())
