"""Classical synchronization baseline on two real DC22 observation aliases.

Search only an unchanged author-learned program. Validate candidate actions in
the downloaded, unmodified game through the author's adapter. This is a small
baseline/feasibility test, not a new algorithm, learned experiment constructor,
attack, or complete task evaluation. The learned program is in-sample.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from copy import deepcopy
import gzip
import importlib.metadata
import json
from pathlib import Path
import time

from screen_opine_observation_aliases import decode_steps, save
from screen_opine_query_memory import (
    ROOT, VENDOR, ARCHIVE, action, grid, import_path, memory_hash,
    mismatch, restore, sha, snapshot,
)


NATIVE = ROOT / "vendor/arc_public_envs/dc22/4c9bff3e/dc22.py"
MODEL = ROOT / "runs/opine_query_memory_v0/round_086/game_engine.py"
STARTS = (762, 766)


def load_native_adapter():
    play = import_path(VENDOR / "play.py", "_opine_play_sync")
    play._register_pkg_stubs()
    play._load("src.opine_world.synth_loop.domain_adapter",
               VENDOR / "src/opine_world/synth_loop/domain_adapter.py")
    adapter = play._load("src.opine_world.synth_loop.adapters.arcengine_adapter",
                        VENDOR / "src/opine_world/synth_loop/adapters/arcengine_adapter.py")
    cls = play._load_game_class("dc22", NATIVE)
    return adapter.ArcEngineEnv(cls())


def frame_bytes(frame):
    return bytes(int(value) for row in frame for value in row)


def frame_key(frame, mask):
    return bytes(frame[r][c] for r in range(64) for c in range(64)
                 if (r, c) not in mask)


def state_key(state, mask):
    frame, memory = state
    return frame_key(frame, mask), memory_hash(memory)


def native_states(steps, out):
    env = load_native_adapter()
    env.reset()
    states, records = {}, []
    for step in range(max(STARTS)):
        if step in STARTS:
            states[step] = deepcopy(env)
        act = action(steps[step]["action"])
        _, reward, done = env.step(act)
        actual = env.get_frame().tolist()
        expected = grid(steps[step]["frame"])
        differences = mismatch(actual, expected, set())
        record = {"step": step, "action": act, "different_pixels": differences,
                  "level_after": env.game.level_index, "reward": reward, "done": done,
                  "frame_sha256": sha(frame_bytes(actual))}
        records.append(record)
        if differences:
            save(out / "native_replay.json", records)
            save(out / "first_mismatch.json", {"step": step, "actual": actual, "expected": expected})
            return None, {"status": "replay_mismatch", **record,
                          "scope": "No search or native outcome claim: public game/version does not exactly match this archive prefix."}
    states[max(STARTS)] = deepcopy(env)
    save(out / "native_replay.json", records)
    return states, {"status": "exact", "actions": len(records),
                    "all_pixels_match": True, "starts": list(STARTS)}


def learned_states(steps, planner):
    model = planner.load_model(MODEL, module_name="_dc22_sync_learned")
    names = sorted({name for node in ast.walk(ast.parse(MODEL.read_text()))
                    if isinstance(node, ast.Global) for name in node.names})
    states = {}
    for step in range(659, max(STARTS)+1):
        before = grid(steps[step-1]["frame"])
        if step in STARTS:
            states[step] = (before, snapshot(model.module, names))
        predicted, _, _ = planner.simulate_step(model, before, action(steps[step]["action"]))
        assert mismatch(predicted, grid(steps[step]["frame"]), model.move_counter_mask) == 0
    return model, names, tuple(states[s] for s in STARTS)


def search(model, names, states, actions, planner, *, max_depth, max_nodes, timeout_s):
    mask = set(model.move_counter_mask)
    key = lambda pair: tuple(state_key(state, mask) for state in pair)
    initial_key = key(states)
    assert initial_key[0][0] == initial_key[1][0]
    assert initial_key[0][1] != initial_key[1][1]
    queue, seen = deque([(states, [])]), {initial_key}
    started, expanded, simulated = time.monotonic(), 0, 0
    while queue:
        if time.monotonic()-started > timeout_s or expanded >= max_nodes:
            return {"status": "budget_exhausted", "expanded": expanded,
                    "simulated_transitions": simulated, "seen": len(seen),
                    "elapsed_s": time.monotonic()-started}
        pair, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        expanded += 1
        for act in actions:
            children = []
            terminal = False
            for before, memory in pair:
                restore(model.module, memory)
                predicted, reward, done = planner.simulate_step(model, before, act)
                simulated += 1
                terminal |= reward > 0 or done
                children.append((predicted, snapshot(model.module, names)))
            if terminal:
                continue
            child_key = key(children)
            if child_key[0] == child_key[1]:
                return {"status": "found", "plan": path+[act], "expanded": expanded,
                        "simulated_transitions": simulated, "seen": len(seen),
                        "elapsed_s": time.monotonic()-started}
            if child_key not in seen:
                seen.add(child_key)
                queue.append((children, path+[act]))
    return {"status": "no_plan_within_depth", "expanded": expanded,
            "simulated_transitions": simulated, "seen": len(seen),
            "elapsed_s": time.monotonic()-started}


def validate(states, plan, probes):
    outputs, paths = [], []
    for step in STARTS:
        env = deepcopy(states[step])
        trace = []
        for act in plan:
            _, reward, done = env.step(act)
            trace.append({"action": act, "reward": reward, "done": done,
                          "level_after": env.game.level_index,
                          "game_over": env.is_game_over(),
                          "frame_sha256": sha(frame_bytes(env.get_frame()))})
        outputs.append(env)
        paths.append({"start": step, "steps": trace})
    # Only HUD row 63 is omitted, matching the original learned-model mask.
    mask = {(63, c) for c in range(64)}
    difference = mismatch(outputs[0].get_frame().tolist(), outputs[1].get_frame().tolist(), mask)
    probe_results = []
    for act in probes:
        futures = []
        for env in outputs:
            fork = deepcopy(env)
            _, reward, done = fork.step(act)
            futures.append((fork.get_frame().tolist(), reward, done))
        probe_results.append({"action": act,
                              "different_scene_pixels": mismatch(futures[0][0], futures[1][0], mask),
                              "reward_done_agree": futures[0][1:] == futures[1][1:]})
    return {"plan": plan, "paths": paths, "final_scene_difference": difference,
            "one_step_probes": probe_results,
            "one_step_disagreements": sum(p['different_scene_pixels'] > 0 or
                                           not p['reward_done_agree'] for p in probe_results),
            "evaluation_actions": 2*len(plan)+2*len(probes),
            "scope": "Two archived native starts and this finite probe set only. Equal frames or one-step probes do not prove hidden-state equality or arbitrary-horizon synchronization."}


def main(args):
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(gzip.decompress(ARCHIVE.read_bytes()))
    steps, _ = decode_steps(data)
    labels = sorted({steps[t]["action"] for t in range(659, max(STARTS)+1)})
    actions = [action(label) for label in labels if label != "RESET"]
    save(out / "plan.json", {
        "scope": __doc__, "source_starts": list(STARTS), "actions": actions,
        "max_depth": args.max_depth, "max_nodes": args.max_nodes,
        "timeout_s": args.timeout_s, "api_requests": 0, "gpu_work": False,
        "native_game_sha256": sha(NATIVE.read_bytes()),
        "learned_model_sha256": sha(MODEL.read_bytes()),
        "archive_sha256": sha(ARCHIVE.read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "versions": {p: importlib.metadata.version(p) for p in ("arc-agi", "arcengine", "numpy")},
        "baseline": "Ordinary breadth-first search in the product of two learned simulator states. No task completion or novelty credit is assigned.",
        "information_boundary": "Search sees the published learned program and frame/action archive; native source and native hidden state are used only by the evaluator. The model had already seen the archive during author synthesis."})
    native, replay = native_states(steps, out)
    if native is None:
        result = {"status": "stopped", "replay": replay, "elapsed_s": time.monotonic()-started,
                  "api_requests": 0, "gpu_work": False}
        save(out / "summary.json", result)
        print(json.dumps(result), flush=True)
        return
    print(json.dumps({"phase": "native_replay", **replay}), flush=True)
    planner = import_path(VENDOR / "src/opine_world/synth_loop/planner.py", "_sync_native_planner")
    model, names, initial = learned_states(steps, planner)
    found = search(model, names, initial, actions, planner,
                   max_depth=args.max_depth, max_nodes=args.max_nodes, timeout_s=args.timeout_s)
    save(out / "search.json", found)
    controls = {"no_action": validate(native, [], actions),
                "level_reset": validate(native, [0], actions)}
    if found["status"] == "found":
        controls["searched_sequence"] = validate(native, found["plan"], actions)
    save(out / "native_validation.json", controls)
    result = {"status": "completed", "replay": replay, "search": found,
              "native_validation": {name: {k:v for k,v in value.items()
                                            if k not in ("paths", "one_step_probes")}
                                    for name, value in controls.items()},
              "elapsed_s": time.monotonic()-started, "api_requests": 0, "gpu_work": False,
              "novelty": "Classical synchronization baseline only; no new learning operator established."}
    save(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=30)
    main(parser.parse_args())
