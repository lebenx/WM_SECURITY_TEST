"""One fresh-module native OPINE planner call at an independently qualified root.

No retrospective factual-memory initialization, added actions or fallback actor.
The parent process bounds wall time without replacing native planner SIGALRM.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np

from probe_opine_level_reset_replay import ROOT, actual_frame, load_native, native_counters

DEFAULT_CONFIG = ROOT / "configs/opine_native_clean_root_20261015.json"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_planner(path):
    spec = importlib.util.spec_from_file_location("_opine_clean_planner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_worker(config, output):
    started = time.monotonic()
    summary = {
        "status": "running_resource_gate", "resource_qualified": False,
        "planner": None, "actual_return": None, "actual_task_success": None,
        "controls_executed": 0, "controls_started": 0,
        "activity": {"planner_calls": 0, "learned_program_imports": 0,
                     "transition_calls": None, "reward_calls": None,
                     "api_calls": 0, "gpu_calls": 0, "training_updates": 0,
                     "attack_queries": 0, "explicit_env_resets": 0,
                     "registration_env_constructions": 0},
        "scope": config["interpretation"],
    }
    frames, rows = [], []
    model = None
    counts = {"transition_calls": 0, "reward_calls": 0}
    memory_names = []

    def checkpoint():
        summary["elapsed_s"] = time.monotonic() - started
        write_json(output / "summary.json", summary)

    def deadline():
        if time.monotonic() - started >= config["limits"]["wallclock_s"]:
            raise TimeoutError("Total Clean probe wall-clock limit exhausted")

    checkpoint()
    try:
        resource = ROOT / config["resource_run"]
        resource_summary = read_json(resource / "summary.json")
        metadata = read_json(resource / config["root_metadata"])
        if not (resource_summary.get("status") == "completed_native_reset_replay"
                and resource_summary.get("qualification") is True
                and metadata.get("status") == "qualified_actual_native_root"
                and metadata.get("qualified_by_completed_panel") is True
                and metadata.get("before_action_index") == config["before_action_index"]
                and metadata.get("fullframe_difference_from_archive") == 0
                and metadata["native_counters"]["level_index"] == config["level_index"]):
            raise ValueError("Resource replay/root qualification is absent or failed")
        summary["packages"] = {name: importlib.metadata.version(name)
                               for name in config["runtime"]["required_versions"]}
        if summary["packages"] != config["runtime"]["required_versions"]:
            raise ValueError("Runtime differs from the registered resource runtime")
        # Construct only to register the same local classes; never reset/step it.
        registration_env = load_native(read_json(resource / "protocol.json"))
        summary["activity"]["registration_env_constructions"] = 1
        del registration_env
        with (resource / config["snapshot"]).open("rb") as handle:
            root = pickle.load(handle)
        if root["before_action_index"] != config["before_action_index"]:
            raise ValueError("Unexpected saved action index")
        env = deepcopy(root["env"])
        if env is root["env"] or env.game is root["env"].game:
            raise ValueError("Execution environment aliases the loaded root")
        for candidate in (root["env"], env):
            if not np.array_equal(actual_frame(candidate), root["actual_frame"]):
                raise ValueError("Loaded/deep-copied frame differs from actual snapshot")
            if native_counters(candidate) != metadata["native_counters"]:
                raise ValueError("Loaded/deep-copied counters differ from snapshot")
        random.setstate(root["python_random_state"])
        np.random.set_state(root["numpy_random_state"])
        frame = actual_frame(env).tolist()
        actions = env.get_available_actions()
        initial_level = int(env.game.level_index)
        if env.is_game_over() or env.is_game_won():
            raise ValueError("The qualified root is already terminal")
        summary.update(resource_qualified=True,
                       initial_native_counters=native_counters(env),
                       available_actions=actions,
                       goal="Native L6 completion: level advance or game win")
        np.save(output / "initial_frame.npy", np.asarray(frame, dtype=np.int8))
        deadline()

        planner_path = ROOT / config["planner_source"]
        model_path = ROOT / config["model_source"]
        planner = load_planner(planner_path)
        model = planner.load_model(model_path, module_name="_opine_clean_round086_fresh")
        summary["activity"]["learned_program_imports"] = 1
        (output / "published_game_engine.py").write_bytes(model_path.read_bytes())
        (output / "native_planner.py").write_bytes(planner_path.read_bytes())
        memory_names = sorted({name for node in ast.walk(ast.parse(model_path.read_text()))
                               if isinstance(node, ast.Global) for name in node.names})
        initial_memory = {name: deepcopy(getattr(model.module, name)) for name in memory_names}
        with (output / "memory_initial.pkl").open("wb") as handle:
            pickle.dump(initial_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        summary["declared_global_names"] = memory_names
        summary["memory_initialization"] = "Fresh module only; no transition calls before planning"

        def counted(function, key):
            def wrapped(*args, **kwargs):
                counts[key] += 1
                return function(*args, **kwargs)
            return wrapped

        model.transition_function = counted(model.transition_function, "transition_calls")
        model.reward_function = counted(model.reward_function, "reward_calls")
        model.module.transition_function = model.transition_function
        model.module.reward_function = model.reward_function
        summary["status"] = "running_native_planner"
        summary["activity"]["planner_calls"] = 1
        checkpoint()
        call_started = time.monotonic()
        result = planner.plan_from_model(
            model, frame, actions, frames_only=True, max_depth=0,
            max_nodes=0, timeout_s=30)
        summary["planner_elapsed_s"] = time.monotonic() - call_started
        summary["planner"] = result.summary()
        summary["activity"].update(counts)
        write_json(output / "plan.json", {
            **result.summary(), "plan": result.plan,
            "predicted_steps": [step.to_dict() for step in result.predicted_steps]})
        deadline()
        if not result.ok:
            status = {"timeout": "planner_timeout", "model_error": "planner_model_error"}
            summary["status"] = status.get(result.source, "planner_no_plan")
            if result.plan:
                summary["status"] = "planner_rejected_plan"
            return 0
        if not 1 <= len(result.plan) <= config["real_execution"]["max_controls"]:
            raise ValueError("Native returned plan violates the fixed execution bound")

        summary.update(status="executing_native_plan", actual_return=0.0)
        with (output / "execution.jsonl").open("x") as trace:
            for index, action in enumerate(result.plan):
                deadline()
                summary["controls_started"] += 1
                checkpoint()
                _, reward, done = env.step(action)
                actual = actual_frame(env)
                row = {"control_index": index, "action": action,
                       "reward": float(reward), "done": bool(done),
                       "level_advanced": int(env.game.level_index) > initial_level,
                       "game_won": bool(env.is_game_won()),
                       "game_over": bool(env.is_game_over()),
                       "native_counters": native_counters(env)}
                rows.append(row)
                frames.append(actual)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                summary["controls_executed"] += 1
                summary["actual_return"] += float(reward)
                summary["last_actual_outcome"] = row
                checkpoint()
                deadline()
                if row["level_advanced"] or row["game_won"] or row["game_over"] or done:
                    break
        summary.update(status="completed_plan_execution",
                       actual_task_success=any(r["level_advanced"] or r["game_won"] for r in rows),
                       stopping_reason=("native_terminal_or_level_advance" if rows and (
                           rows[-1]["done"] or rows[-1]["game_won"] or rows[-1]["game_over"]
                           or rows[-1]["level_advanced"]) else "returned_plan_exhausted"))
        return 0
    except TimeoutError as exc:
        summary.update(status="partial_resource_deferred", exception=str(exc))
        return 1
    except BaseException as exc:
        summary.update(status="technical_error", exception_type=type(exc).__name__,
                       exception=str(exc))
        return 1
    finally:
        if model is not None:
            summary["activity"].update(counts)
            with (output / "memory_final.pkl").open("wb") as handle:
                pickle.dump({name: deepcopy(getattr(model.module, name)) for name in memory_names},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(output / "actual_execution_frames.npz",
                            frames=np.stack(frames) if frames else np.empty((0, 64, 64), dtype=np.int8))
        checkpoint()
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config["status"] != "approved_for_one_cpu_clean":
        raise ValueError("Protocol needs root review before execution")
    if (config["before_action_index"], config["planner"]["calls"],
        config["planner"]["max_depth"], config["planner"]["max_nodes"],
        config["planner"]["timeout_s"], config["real_execution"]["max_controls"],
        config["limits"]["wallclock_s"]) != (762, 1, 0, 0, 30, 40, 60):
        raise ValueError("Runner supports only the frozen one-call protocol")
    output = (args.output or ROOT / config["default_output"]).resolve()
    if args.worker:
        return run_worker(config, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", config)
    (output / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config",
               str(args.config.resolve()), "--output", str(output)]
    with (output / "worker.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=config["limits"]["wallclock_s"])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            summary_path = output / "summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            summary.update(status="partial_resource_deferred", parent_wallclock_timeout=True,
                           exception="Worker exceeded60seconds; no retry or budget change",
                           partial_record_limit="Counts/outcomes are the last completed saved records; a started control may be interrupted.")
            write_json(summary_path, summary)
            code = 1
    print(json.dumps({"worker_exit_code": code, "output": str(output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
