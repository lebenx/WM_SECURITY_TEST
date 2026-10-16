"""Replay released learned code and test whether imaginary calls alter memory.

This is a diagnostic on author-published, in-sample observations, not an ARC
environment rollout or an attack evaluation. Programs are extracted unchanged.
Memory is initialized by replaying actual level-six observations and actions;
no latent value is chosen to make a probe succeed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

from screen_opine_observation_aliases import decode_steps, save


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/opine-world"
ARCHIVE = VENDOR / "docs/replay_data/dc22.json.gz"
WITNESSES = (762, 766, 778, 784)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def grid(value):
    return [list(value[i:i+64]) for i in range(0, 4096, 64)]


def action(value):
    if value == "RESET":
        return 0
    if value.startswith("ACTION"):
        return int(value[6:])
    match = re.fullmatch(r"click\((\d+),(\d+)\)", value)
    assert match, value
    return {"action_id": 6, "x": int(match[1]), "y": int(match[2])}


def import_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(module, names):
    return {name: deepcopy(getattr(module, name)) for name in names}


def restore(module, values):
    for name, value in values.items():
        setattr(module, name, deepcopy(value))


def memory_hash(values):
    return sha(json.dumps(values, sort_keys=True).encode())


def mismatch(a, b, mask):
    assert len(a) == len(b) == 64
    assert all(len(row) == 64 for row in a + b)
    return sum(a[r][c] != b[r][c] for r in range(64) for c in range(64)
               if (r, c) not in mask)


def extract_codes(data):
    lines, codes = [], []
    for engine in data["engines"]:
        if "t" in engine:
            lines = engine["t"].split("\n")
        else:
            for start, count, added in reversed(engine["o"]):
                lines[start:start+count] = added
        codes.append("\n".join(lines))
    return codes


def screen_round(out, data, steps, codes, rnd, native_planner):
    event = next(event for event in data["synth"] if event["round"] == rnd)
    end = data["frames"][event["f"]]["s"]
    # First observed entry frame for L6 follows completion of L5 at step 658.
    entry = 658
    assert steps[entry+1]["level"] == 5
    assert all(steps[t]["level"] == 5 for t in range(entry+1, end+1))
    folder = out / f"round_{rnd:03d}"
    folder.mkdir()
    code = codes[event["e"]]
    code_path = folder / "game_engine.py"
    code_path.write_text(code)
    with (folder / "l6_initial.pkl").open("wb") as file:
        pickle.dump(grid(steps[entry]["frame"]), file)
    model = native_planner.load_model(code_path, module_name=f"_dc22_memory_{rnd}")
    module = model.module
    tree = ast.parse(code)
    names = sorted({name for node in ast.walk(tree) if isinstance(node, ast.Global)
                    for name in node.names})
    initial = snapshot(module, names)
    mask = set(model.move_counter_mask)
    records, planner_records = [], []
    for step in range(entry+1, end+1):
        before, after = grid(steps[step-1]["frame"]), grid(steps[step]["frame"])
        act = action(steps[step]["action"])
        original_input = deepcopy(before)
        factual_memory = snapshot(module, names)
        clean, _, _ = native_planner.simulate_step(model, before, act)
        next_memory = snapshot(module, names)
        repeated, _, _ = native_planner.simulate_step(model, before, act)
        repeated_memory = snapshot(module, names)
        restore(module, factual_memory)
        native_planner.simulate_step(model, before, {"action_id": 6, "x": 0, "y": 0})
        noop, _, _ = native_planner.simulate_step(model, before, act)
        restore(module, initial)
        cold, _, _ = native_planner.simulate_step(model, before, act)
        restore(module, factual_memory)
        recovered, _, _ = native_planner.simulate_step(model, before, act)
        assert recovered == clean
        assert before == original_input
        record = {
            "step": step, "action": act,
            "clean_mismatched_scene_pixels": mismatch(clean, after, mask),
            "repeated_mismatched_scene_pixels": mismatch(repeated, after, mask),
            "repeat_vs_clean_scene_pixels": mismatch(repeated, clean, mask),
            "noop_vs_clean_scene_pixels": mismatch(noop, clean, mask),
            "cold_mismatched_scene_pixels": mismatch(cold, after, mask),
            "memory_changed_by_first_query": [n for n in names if factual_memory[n] != next_memory[n]],
            "memory_changed_by_second_query": [n for n in names if next_memory[n] != repeated_memory[n]],
            "factual_memory_sha256": memory_hash(factual_memory),
            "input_unchanged": True, "restoration_exact": True,
        }
        if step in WITNESSES:
            record.update(before_sha256=sha(steps[step-1]["frame"]),
                          after_sha256=sha(steps[step]["frame"]))
            # Test the actual generated planner through the upstream entry
            # point, without adding missing actions to its candidate set.
            for mode, start_memory in (("factual_prefix", factual_memory),
                                       ("fresh_module", initial)):
                restore(module, start_memory)
                started = time.monotonic()
                result = native_planner.plan_from_model(
                    model, before, [1, 2, 3, 4, 6], frames_only=True,
                    max_depth=4, timeout_s=3)
                planning_memory = snapshot(module, names)
                predicted, _, _ = native_planner.simulate_step(model, before, act)
                restore(module, start_memory)
                without_plan, _, _ = native_planner.simulate_step(model, before, act)
                planner_records.append({
                    "step": step, "initialization": mode, **result.summary(),
                    "elapsed_s": time.monotonic()-started,
                    "changed_memory_names": [n for n in names if planning_memory[n] != start_memory[n]],
                    "post_plan_vs_pre_plan_scene_pixels": mismatch(predicted, without_plan, mask),
                    "native_candidate_actions": module._planner_actions(before, [1, 2, 3, 4, 6]),
                })
        records.append(record)
        # Imaginary probes have no influence on factual chronological replay.
        restore(module, next_memory)
    save(folder / "transitions.json", records)
    save(folder / "planner_probes.json", planner_records)
    result = {
        "round": rnd, "code_sha256": sha(code.encode()),
        "author_reported_accuracy_percent": event["acc"],
        "observed_prefix_end": end, "transitions": len(records),
        "clean_correct": sum(r["clean_mismatched_scene_pixels"] == 0 for r in records),
        "repeated_correct": sum(r["repeated_mismatched_scene_pixels"] == 0 for r in records),
        "cold_correct": sum(r["cold_mismatched_scene_pixels"] == 0 for r in records),
        "query_order_sensitive": sum(r["repeat_vs_clean_scene_pixels"] > 0 for r in records),
        "clean_correct_to_repeat_wrong": sum(r["clean_mismatched_scene_pixels"] == 0 and
                                            r["repeated_mismatched_scene_pixels"] > 0 for r in records),
        "noop_control_changes": sum(r["noop_vs_clean_scene_pixels"] > 0 for r in records),
        "all_input_and_restoration_checks_passed": True,
        "mutable_global_names": names,
        "selected_witnesses": [r for r in records if r["step"] in WITNESSES],
        "native_planner_queries": planner_records,
    }
    save(folder / "summary.json", result)
    return result


def main(args):
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(gzip.decompress(ARCHIVE.read_bytes()))
    steps, _ = decode_steps(data)
    codes = extract_codes(data)
    save(out / "plan.json", {
        "scope": __doc__, "rounds": [63, 86], "selected_steps": list(WITNESSES),
        "archive_sha256": sha(ARCHIVE.read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "upstream_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "entry_frame_step": 658, "native_planner_probe_max_depth": 4,
        "native_planner_probe_timeout_s": 3, "api_requests": 0, "gpu_work": False,
        "controls": ["repeat identical query without factual action", "out-of-panel click before query",
                     "reset learned globals per query", "restore factual memory before query"],
        "limitation": "Later models saw these data during synthesis. Factual-prefix initialization is a diagnostic, not an assertion that the native planner imports that history. No withheld evaluation, native task outcome, or novelty claim."})
    native_planner = import_path(VENDOR / "src/opine_world/synth_loop/planner.py", "_native_opine_planner")
    results = [screen_round(out, data, steps, codes, rnd, native_planner) for rnd in (63, 86)]
    summary = {"status": "completed", "results": results,
               "elapsed_s": time.monotonic()-started, "api_requests": 0, "gpu_work": False}
    save(out / "summary.json", summary)
    print(json.dumps({**summary, "results": [{k:v for k,v in result.items()
                     if k not in ("selected_witnesses", "native_planner_queries")} for result in results]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
