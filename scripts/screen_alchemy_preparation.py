"""Bounded native Alchemy baseline screen, not a proposed transfer method.

Both policies receive the native belief update, known perceptual mappings and
the full native prior over bottleneck graphs. The strong policy is the upstream
ideal explorer; the weak comparator maximizes its one-step objective. All
transitions run through upstream SymbolicAlchemy. No LLM or GPU is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import copy, deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from dm_alchemy import symbolic_alchemy as sa
from dm_alchemy import symbolic_alchemy_trackers as trackers
from dm_alchemy.ideal_observer import ideal_observer as io
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import stones_and_potions as sp
from dm_alchemy.types import utils


def remaining_worlds(belief):
    return int(np.count_nonzero(belief.world_state_distribution.poss_world_states))


def one_step_objectives(belief, precomputed):
    """Expected remaining world count, using native outcome probabilities."""
    objectives = []
    for stone, potion in belief.possible_actions(precomputed):
        if potion == io.END_TRIAL:
            expected = float(remaining_worlds(belief))
        else:
            outcomes, _ = belief.world_state_distribution.possible_outcomes(
                potion, stone, precomputed)
            expected = 0.0
            total_prob = 0.0
            for outcome in outcomes:
                branch = deepcopy(belief)
                prob = branch.action_and_outcome(
                    stone, potion, outcome, precomputed,
                    io.stone_potion_bit_mask(stone, potion, precomputed))
                expected += float(prob) * remaining_worlds(branch)
                total_prob += float(prob)
            if not np.isclose(total_prob, 1.0):
                raise ValueError(f"Native outcome probability sum: {total_prob}")
        objectives.append({"action": [int(stone), int(potion)],
                           "expected_remaining_worlds": expected})
    return objectives


def run_policy(seed, policy, precomputed, trials, potions):
    env = sa.get_symbolic_alchemy_level(
        "all_fixed_with_random_bottleneck", seed=seed, num_trials=trials,
        num_stones_per_trial=1, num_potions_per_trial=potions,
        max_steps_per_trial=potions + 4, end_trial_action=True)
    tracker = trackers.BeliefStateTracker(precomputed, env)
    env.add_trackers({tracker.NAME: tracker})
    physical_actions = []
    native_step = env.step_no_observation

    def counted_step(action, *native_args, **native_kwargs):
        physical_actions.append({"trial": env.trial_number,
                                 "action": repr(action)})
        return native_step(action, *native_args, **native_kwargs)

    env.step_no_observation = counted_step
    timestep = env.reset()
    search_results = {}
    decisions = []
    trial_end_worlds = []
    score = 0.0
    planning_seconds = 0.0
    native_action_limit = trials * (potions + 6)
    while not timestep.last():
        if len(physical_actions) > native_action_limit:
            raise RuntimeError("Bounded screen exceeded its native action limit")
        belief = tracker.belief_state.belief_state
        before = remaining_worlds(belief)
        trial = env.trial_number
        objectives = one_step_objectives(belief, precomputed)
        start = time.monotonic()
        cached_before = len(search_results)
        if policy == "native_ideal_explorer":
            action, objective, search_results = io.ideal_observer(
                belief, search_results, 12, precomputed, True)
            objective = float(objective)
        else:
            # END_TRIAL appears first upstream and wins exact objective ties.
            best = min(objectives, key=lambda x: round(
                x["expected_remaining_worlds"], 10))
            action = tuple(best["action"])
            objective = -best["expected_remaining_worlds"]
        latency = time.monotonic() - start
        planning_seconds += latency
        key = [int(x) for x in action]
        selected_myopic = next(x["expected_remaining_worlds"]
                               for x in objectives if x["action"] == key)
        record = {
            "trial": trial, "before_worlds": before,
            "stones": {str(k): int(v) for k, v in belief.aligned_stones.items()},
            "potions": {str(k): int(v) for k, v in belief.perceived_potions.items()},
            "one_step_objectives": objectives, "selected_action": key,
            "selected_expected_immediate_reduction": before - selected_myopic,
            "best_expected_immediate_reduction": before - min(
                x["expected_remaining_worlds"] for x in objectives),
            "native_or_myopic_objective": objective,
            "selection_seconds": latency,
            "new_search_cache_entries": len(search_results) - cached_before,
        }
        concrete = utils.type_based_action_from_ints(
            *action, tracker.belief_state.rotation)
        timestep = sa.take_simplified_action(concrete, env)
        score += float(timestep.reward or 0.0)
        record["after_worlds"] = remaining_worlds(tracker.belief_state.belief_state)
        record["reward"] = float(timestep.reward or 0.0)
        record["trial_ended"] = timestep.last() or env.trial_number != trial
        if record["trial_ended"]:
            trial_end_worlds.append(record["after_worlds"])
        decisions.append(record)
    return {
        "seed": seed, "policy": policy, "status": "completed",
        "score": score, "trial_end_worlds": trial_end_worlds,
        "planning_seconds_excluding_shared_one_step_diagnostic": planning_seconds,
        "search_cache_entries": len(search_results),
        "native_actions": len(physical_actions), "decisions": decisions,
        "physical_actions": physical_actions,
    }


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: (output / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    started = time.monotonic()
    commit = subprocess.check_output(
        ["git", "-C", "vendor/dm_alchemy", "rev-parse", "HEAD"], text=True).strip()
    native_file = Path(io.__file__)
    vendor_file = Path("vendor/dm_alchemy/dm_alchemy/ideal_observer/ideal_observer.py")
    native_hash = hashlib.sha256(native_file.read_bytes()).hexdigest()
    assert native_hash == hashlib.sha256(vendor_file.read_bytes()).hexdigest()
    save("plan.json", vars(args) | {
        "upstream": "https://github.com/google-deepmind/dm_alchemy",
        "commit": commit, "native_planner_sha256": native_hash,
        "policies": ["myopic_world_count", "native_ideal_explorer"],
        "hypothesis": "Native lookahead may already supply informative-test preparation.",
        "scope": "Small baseline feasibility screen; no learned transfer method, held-out evaluation or full benchmark claim.",
        "prior": "Known identity stone/potion maps; unmodified native random-bottleneck graph prior.",
        "versions": {p: importlib.metadata.version(p) for p in
                     ("dm-alchemy", "dm-env-rpc", "numpy", "scipy", "protobuf")},
        "api_requests": 0, "gpu_seconds": 0,
    })
    pre = copy(precomputed_maps.load_from_level_name(
        "perceptual_mapping_randomized_with_random_bottleneck"))
    # Select the known mappings in the prior; transition/search code is unchanged.
    pre.stone_maps = np.array([sp.all_fixed_stone_map().index()])
    pre.potion_maps = np.array([
        sp.all_fixed_potion_map().index(pre.perm_index_to_index)])
    results = []
    for seed in args.seeds:
        for policy in ("myopic_world_count", "native_ideal_explorer"):
            result = run_policy(seed, policy, pre, args.trials, args.potions)
            results.append(result)
            save(f"seed{seed}_{policy}.json", result)
            print(json.dumps({k: result[k] for k in
                              ("seed", "policy", "score", "trial_end_worlds", "native_actions")}), flush=True)
    full = [r for r in results if r["policy"] == "native_ideal_explorer"]
    prep = [{"seed": r["seed"], "decision": i, **d}
            for r in full for i, d in enumerate(r["decisions"])
            if d["selected_action"][1] != io.END_TRIAL
            and abs(d["selected_expected_immediate_reduction"]) < 1e-9
            and -d["native_or_myopic_objective"] < d["before_worlds"] - 1e-9]
    summary = {
        "status": "completed", "seeds": args.seeds,
        "episodes": len(results), "trials_per_episode": args.trials,
        "native_actions": sum(r["native_actions"] for r in results),
        "api_requests": 0, "gpu_seconds": 0,
        "elapsed_s": time.monotonic() - started,
        "zero_immediate_information_preparation_decisions": prep,
        "paired_results": [{"seed": seed, **{
            r["policy"]: {k: r[k] for k in (
                "score", "trial_end_worlds", "native_actions", "search_cache_entries",
                "planning_seconds_excluding_shared_one_step_diagnostic")}
            for r in results if r["seed"] == seed}} for seed in args.seeds],
        "accepted_novel_method": False,
        "interpretation": "Preparation found in the existing planner is a baseline capability, not support for a novel transfer operation.",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save("summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in (
        "zero_immediate_information_preparation_decisions", "paired_results")}), flush=True)
    print(json.dumps({"preparation_decisions": len(prep)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--potions", type=int, default=4)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
