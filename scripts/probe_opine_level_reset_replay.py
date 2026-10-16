"""One bounded native replay; qualify real L6 roots after a natural level reset.

No learned program, planner, model, API, or renderer modification is used.
The previous full-prefix equality failure remains a separate failed gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import copyreg
import gzip
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import pickle
import random
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/opine_level_reset_replay_snapshotfix_20261015.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def decode_archive(path):
    data = json.loads(gzip.decompress(path.read_bytes()))
    current = np.zeros((64, 64), dtype=np.int8)
    steps = {}
    for frame in data["frames"]:
        if "g" in frame:
            current[:] = np.asarray(frame["g"], dtype=np.int8)
        for x, y, value in frame.get("d", []):
            current[y, x] = value
        record = steps.setdefault(frame["s"], {})
        if "a" in frame:
            if "action" in record:
                raise ValueError("Duplicate action label in one archived step")
            record["action"] = frame["a"]
        record["frame"] = current.copy()
    if sorted(steps) != list(range(len(steps))):
        raise ValueError("Noncontiguous archive action indices")
    completions = [data["frames"][i]["s"] for i in data["level_steps"]]
    if completions[4] != 658:
        raise ValueError("Unexpected archive L6 entry")
    return steps, completions


def parse_action(label):
    import re
    if label == "RESET":
        return 0
    if label.startswith("ACTION"):
        return int(label[6:])
    match = re.fullmatch(r"click\((\d+),(\d+)\)", label)
    if match is None:
        raise ValueError(f"Unsupported archived action {label!r}")
    return {"action_id": 6, "x": int(match[1]), "y": int(match[2])}


def load_native(config):
    """Use author imports, registering the unchanged game for local pickle IO."""
    play_path = ROOT / "vendor/opine-world/play.py"
    spec = importlib.util.spec_from_file_location("_opine_reset_replay_play", play_path)
    play = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = play
    spec.loader.exec_module(play)
    play._register_pkg_stubs()
    play._load("src.opine_world.synth_loop.domain_adapter",
               ROOT / "vendor/opine-world/src/opine_world/synth_loop/domain_adapter.py")
    adapter = play._load("src.opine_world.synth_loop.adapters.arcengine_adapter",
                        ROOT / config["adapter"])
    if "dc22" in sys.modules:
        raise RuntimeError("Game module dc22 already loaded; require a fresh process")
    game_module = play._load("dc22", ROOT / config["game_source"])
    return adapter.ArcEngineEnv(getattr(game_module, config["game_class"])())


def actual_frame(env):
    frame = np.asarray(env.get_frame(), dtype=np.int8)
    if frame.shape != (64, 64):
        raise ValueError(f"Invalid native frame shape {frame.shape}")
    return frame.copy()


def differences(actual, expected):
    rc = np.argwhere(actual != expected)
    return [[int(r), int(c), int(expected[r, c]), int(actual[r, c])] for r, c in rc]


def native_counters(env):
    game = env.game
    return {"level_index": int(game.level_index),
            "action_count": int(game._action_count),
            "score": float(game._score),
            "ui_remaining_steps": int(game.step_counter_ui.current_steps),
            "game_state": str(game._state)}


def snapshot_dumps(payload):
    """Preserve native Enum identities without changing any global reducer.

    arcengine 0.9.3 GameAction changes _value_ inside __init__, so its
    default value-based pickle reconstruction raises ValueError. Name-based
    reconstruction returns the exact existing member, as deepcopy already does.
    """
    from arcengine.enums import GameAction

    def reduce_action(member):
        return getattr, (GameAction, member.name)

    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = copyreg.dispatch_table.copy()
    pickler.dispatch_table[GameAction] = reduce_action
    pickler.dump(payload)
    return buffer.getvalue()


def save_root(env, index, expected, out):
    before = actual_frame(env)
    clone = deepcopy(env)
    if clone is env or clone.game is env.game or clone.game.current_level is env.game.current_level:
        raise RuntimeError("Root deepcopy aliases the live environment")
    if not np.array_equal(actual_frame(clone), before):
        raise RuntimeError("Root deepcopy changed the observed frame")
    payload = {"env": clone, "before_action_index": index,
               "python_random_state": random.getstate(),
               "numpy_random_state": np.random.get_state(),
               "actual_frame": before.copy()}
    serialized = snapshot_dumps(payload)
    reloaded = pickle.loads(serialized)
    if not np.array_equal(actual_frame(reloaded["env"]), before):
        raise RuntimeError("Local snapshot pickle round-trip changed the frame")
    if native_counters(reloaded["env"]) != native_counters(env):
        raise RuntimeError("Local snapshot pickle round-trip changed counters")
    name = f"root_before_{index}"
    (out / f"{name}.pkl").write_bytes(serialized)
    info = {"before_action_index": index, "snapshot": f"{name}.pkl",
            "status": "candidate_snapshot_requires_final_qualification",
            "fullframe_difference_from_archive": len(differences(before, expected)),
            "deepcopy_and_pickle_frame_checks": True,
            "serialization": "Local Pickler GameAction name reducer; standard pickle.loads; no global Enum or copyreg mutation.",
            "native_counters": native_counters(env),
            "load_contract": "Register the same unchanged native modules with load_native's import sequence before unpickling. Never load untrusted pickle files."}
    write_json(out / f"{name}.json", info)
    return info


def run(config_path, output):
    config = json.loads(config_path.read_text())
    if config["status"] != "approved_for_one_cpu_replay":
        raise ValueError("Protocol must receive root review before execution")
    execution = config["execution"]
    if (execution["first_action_index"], execution["last_action_index_inclusive"],
        execution["max_actions"], execution["wallclock_limit_s"]) != (0, 784, 785, 60):
        raise ValueError("This runner implements only the fixed 785-action/60s protocol")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", config)
    (output / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    summary = {"status": "running", "qualification": None, "actions_executed": 0,
               "config_path": str(config_path.relative_to(ROOT)),
               "old_fullprefix_gate": "failed_at_index482_preserved",
               "roots": [], "l6_entries_after_action": [], "level_transitions": [],
               "activity": {"model_calls": 0, "learned_program_imports": 0,
                            "planner_calls": 0, "api_calls": 0, "gpu_calls": 0,
                            "native_env_resets": 0, "native_actions": 0}}
    frames, rows, reset_checks = [], [], []
    allowed = {tuple(x) for x in config["known_pre_l6_occlusion"]["coordinates_rc"]}
    pre_unrecognized, post_mismatches = [], []

    def timeout_handler(_signum, _frame):
        raise TimeoutError("Fixed 60-second native replay budget exhausted")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        summary["packages"] = {name: importlib.metadata.version(name)
                               for name in config["runtime"]["required_versions"]}
        if summary["packages"] != config["runtime"]["required_versions"]:
            raise RuntimeError("Native runtime version differs from the fixed protocol")
        steps, completions = decode_archive(ROOT / config["archive"])
        summary["archive_level_completions"] = completions
        env = load_native(config)
        env.reset()
        summary["activity"]["native_env_resets"] = 1
        summary["initial_native_counters"] = native_counters(env)
        write_json(output / "summary.json", summary)
        with (output / "replay.jsonl").open("x") as trace:
            for index in range(785):
                if time.monotonic() - started >= 60:
                    raise TimeoutError("Fixed 60-second native replay budget exhausted")
                if index in execution["root_before_action_indices"]:
                    summary["roots"].append(save_root(env, index, steps[index-1]["frame"], output))
                old_level = int(env.game.level_index)
                # Retain references until comparison, avoiding Python object-ID reuse.
                old_sprites = env.game.current_level.get_sprites()
                act = parse_action(steps[index]["action"])
                _, reward, done = env.step(act)
                summary["actions_executed"] += 1
                summary["activity"]["native_actions"] += 1
                # The unchanged adapter catches Exception internally, including
                # a signal-raised TimeoutError. Enforce the limit again outside it.
                if time.monotonic() - started >= 60:
                    raise TimeoutError("Native step returned after the fixed 60-second limit")
                actual = actual_frame(env)
                frames.append(actual)
                diff = differences(actual, steps[index]["frame"])
                new_level = int(env.game.level_index)
                if new_level != old_level:
                    summary["level_transitions"].append({"after_action": index,
                                                         "before": old_level, "after": new_level})
                if new_level == 5 and old_level != 5:
                    summary["l6_entries_after_action"].append(index)
                    new_sprites = env.game.current_level.get_sprites()
                    new_ids = {id(s) for s in new_sprites}
                    reset_checks.append({"after_action": index,
                        "sprite_objects_replaced": not ({id(s) for s in old_sprites} & new_ids),
                        "player_reference_in_current_level": id(env.game.fdvakicpimr) in new_ids,
                        "target_reference_in_current_level": id(env.game.bqxa) in new_ids,
                        "wbze_references_in_current_level": all(id(s) in new_ids for s in env.game.wbzes),
                        "native_counters": native_counters(env)})
                known = all((r, c) in allowed and expected == 12 and native == 15
                            for r, c, expected, native in diff)
                if index < 658 and diff and not known:
                    pre_unrecognized.append(index)
                if index >= 658 and diff:
                    post_mismatches.append(index)
                row = {"action_index": index, "archive_action": steps[index]["action"],
                       "executed_action": act, "level_before": old_level, "level_after": new_level,
                       "reward": float(reward), "done": bool(done),
                       "fullframe_difference_count": len(diff),
                       "scene_difference_count": sum(r != 63 for r, *_ in diff),
                       "differences_rc_archive_native": diff,
                       "only_registered_occlusion_difference": known,
                       "native_counters": native_counters(env)}
                rows.append(row)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if (index+1) % 100 == 0:
                    summary["elapsed_s"] = time.monotonic()-started
                    write_json(output / "summary.json", summary)
                    print(json.dumps({"phase": "native_replay", "actions": index+1,
                                      "level": new_level, "fullframe_difference": len(diff)}), flush=True)
        if time.monotonic() - started >= 60:
            raise TimeoutError("Replay qualification exceeded the fixed 60-second limit")
        checks = {"all_actions_completed": len(rows) == 785,
                  "l6_entry_matches": summary["l6_entries_after_action"] == [658],
                  "l6_fullframe_prefix_exact": not post_mismatches,
                  "all_actual_roots_saved": [r["before_action_index"] for r in summary["roots"]]
                      == execution["root_before_action_indices"],
                  "all_root_frames_and_levels_match": all(
                      r["fullframe_difference_from_archive"] == 0 and
                      r["native_counters"]["level_index"] == 5 for r in summary["roots"]),
                  "native_reset_object_checks": len(reset_checks) == 1 and all(
                      reset_checks[0][key] for key in ["sprite_objects_replaced",
                      "player_reference_in_current_level", "target_reference_in_current_level",
                      "wbze_references_in_current_level"])}
        summary.update(status="completed_native_reset_replay", checks=checks,
                       qualification=all(checks.values()))
    except TimeoutError as exc:
        summary.update(status="partial_resource_deferred", qualification=None,
                       exception_type=type(exc).__name__, exception=str(exc),
                       traceback=traceback.format_exc())
    except BaseException as exc:
        summary.update(status="technical_error", qualification=None,
                       exception_type=type(exc).__name__, exception=str(exc),
                       traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        summary.update(elapsed_s=time.monotonic()-started,
                       pre_l6_unregistered_difference_steps=pre_unrecognized,
                       pre_l6_only_fixed_occlusion_diagnostic=not pre_unrecognized,
                       l6_difference_steps=post_mismatches, reset_checks=reset_checks,
                       fullframe_mismatch_steps=[r["action_index"] for r in rows
                                                 if r["fullframe_difference_count"]],
                       scope="Actual native roots only; no author hidden-state equivalence, whole-prefix equality, or native learned-model factual-memory assimilation claimed.")
        np.savez_compressed(output / "actual_frames.npz",
                            action_indices=np.asarray([r["action_index"] for r in rows]),
                            frames=np.stack(frames[:len(rows)]) if rows else np.empty((0,64,64), dtype=np.int8))
        for root in summary["roots"]:
            root["qualified_by_completed_panel"] = summary["qualification"]
            root["status"] = "qualified_actual_native_root" if summary["qualification"] else "unqualified_or_incomplete"
            write_json(output / f"root_before_{root['before_action_index']}.json", root)
        write_json(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ["status", "qualification", "actions_executed", "elapsed_s"]}), flush=True)
    return 0 if summary["status"] == "completed_native_reset_replay" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/opine_level_reset_replay_snapshotfix_20261015")
    args = parser.parse_args()
    raise SystemExit(run(args.config.resolve(), args.output.resolve()))
