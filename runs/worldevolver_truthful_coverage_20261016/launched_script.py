"""Frozen oracle-coverage qualification; --prepare never imports an API client.

One ScienceWorld JVM replays variations 0/1 at a mechanical root. Real branch
utility is a candidate followed by one fixed continuation, not autonomous
full-task utility. --execute requires a qualified, immutable prepare record.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import re
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/worldevolver_truthful_coverage_20261016.json"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def original_foresight_formatter():
    """Compile only the original two pure static methods, avoiding task imports."""
    path = ROOT / "vendor/WorldEvolver/src/tasks/new/common.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name == "CLSWMTaskMixin")
    names = {"_strip_prediction_prefix", "format_action_conditioned_foresight"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    code = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = {"Any": object}
    exec(compile(code, str(path), "exec"), ns)
    return ns["CLSWMTaskMixin"].format_action_conditioned_foresight


def alarm_handler(signum, frame):
    raise TimeoutError("Frozen phase wall-clock limit reached")


def root_spec(gold, paints, goal):
    idx = next(i for i, a in enumerate(gold) if a.startswith("pour "))
    action = gold[idx]
    delimiter = " into " if " into " in action else " in "
    source, destination = action.rsplit(delimiter, 1)
    present = [p for p in paints if p in source]
    if len(present) != 1:
        raise ValueError("First gold pour does not name exactly one primary paint")
    target = re.search(r"create (.+? paint)\.", goal)
    if not target:
        raise ValueError("Original task target is not readable")
    candidates = sorted([action.replace(present[0], p) for p in paints if p != present[0]]
                        + ["mix " + destination])
    if len(set(candidates)) != 3:
        raise ValueError("Frozen candidate construction did not yield three actions")
    return {"prefix": gold[:idx + 1], "first_pour": action,
            "destination": destination, "target": target.group(1),
            "candidates": candidates,
            "continuation": ["mix " + destination, "look around",
                             "focus on " + target.group(1)]}


class PhysicalBudget:
    def __init__(self, env, cap):
        self.env, self.cap = env, cap
        self.explicit_steps = 0
        self.gold_generator_actions = 0

    @property
    def total(self):
        return self.explicit_steps + self.gold_generator_actions

    def step(self, action):
        if self.total >= self.cap:
            raise RuntimeError("Frozen environment-control budget exhausted")
        self.explicit_steps += 1
        obs, reward, done, info = self.env.step(action)
        return {"action": action, "observation": obs, "reward": reward,
                "done": bool(done), "score": info["score"]}


def parser_error(obs):
    return bool(re.search(r"no known action matches|ambiguous|which one do you mean|"
                          r"i do not understand|i don't understand", obs, re.I))


def physical_root(env, budget, cfg, variation, spec):
    # Native reset regenerates gold when its persistent Java flag is true.
    # Explicit load(False) plus the same initial look avoids repeated gold runs.
    env.load(cfg["task"], variation, cfg["simplifications"], generateGoldPath=False)
    initial = budget.step("look around")
    history = []
    for action in spec["prefix"]:
        record = budget.step(action)
        if record["done"] or parser_error(record["observation"]):
            raise RuntimeError("Official root prefix failed to replay")
        history.append(record)
    return {"initial_observation": initial["observation"], "history": history,
            "goal": env.get_task_description(), "look": env.look(),
            "inventory": env.inventory(), "score": history[-1]["score"],
            "object_tree": env.getObjectTree()}


def prepare(cfg, output):
    from scienceworld import ScienceWorldEnv

    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", cfg)
    (output / "launched_script.py").write_text(Path(__file__).read_text())
    start = time.monotonic()
    summary = {"status": "running", "model_api_calls": 0, "gpu_jobs": 0,
               "roots": [], "scope": cfg["primary_estimand"], "qualified": False}
    save(output / "prepare_summary.json", summary)
    env = None
    budget = None
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(cfg["prepare_wall_seconds"])
    try:
        env = ScienceWorldEnv("", envStepLimit=cfg["env_episode_limit"])
        budget = PhysicalBudget(env, cfg["prepare_environment_control_cap"])
        for variation in cfg["variations"]:
            if variation == 0:
                cache = json.loads((ROOT / cfg["gold_cache_variation0"]).read_text())
                if cache["task"] != cfg["task"] or cache["variation"] != 0:
                    raise ValueError("Official cached gold provenance mismatch")
                gold = [r["action"] for r in cache["records"]]
                goal = cache["goal"]
                gold_source = cfg["gold_cache_variation0"]
            else:
                env.load(cfg["task"], variation, cfg["simplifications"], generateGoldPath=True)
                gold = env.get_gold_action_sequence()
                budget.gold_generator_actions += len(gold)
                if budget.total > budget.cap:
                    raise RuntimeError("Official gold generation exceeded control budget")
                goal = env.get_task_description()
                gold_source = "native_get_gold_action_sequence"
            spec = root_spec(gold, cfg["primary_paints"], goal)
            directory = output / f"variation_{variation}"
            directory.mkdir()
            save(directory / "root_spec.json", spec | {"variation": variation, "goal": goal,
                                                       "gold_source": gold_source, "gold": gold})
            reference = None
            branches = []
            for candidate in spec["candidates"]:
                root = physical_root(env, budget, cfg, variation, spec)
                if reference is None:
                    reference = root
                    save(directory / "root.json", root)
                elif root != reference:
                    raise RuntimeError("Full physical root differs across branch replay")
                first = budget.step(candidate)
                if parser_error(first["observation"]):
                    raise RuntimeError("Frozen candidate is not unambiguously executable")
                if candidate.startswith("pour ") and not first["observation"].startswith("You pour the contents"):
                    raise RuntimeError("Frozen pour candidate did not execute a pour")
                continuation = []
                last = first
                if not first["done"]:
                    for action in spec["continuation"]:
                        last = budget.step(action)
                        continuation.append(last)
                        # An unavailable target after an unsuccessful mixture is a
                        # genuine fixed-controller consequence, not a retry reason.
                        if last["done"]:
                            break
                branches.append({"action": candidate, "first_step": first,
                                 "continuation": continuation,
                                 "utility_score": last["score"],
                                 "success": last["score"] == 100})
                save(directory / "branches.json", branches)
            replay = physical_root(env, budget, cfg, variation, spec)
            exact = replay == reference
            save(directory / "reproduction.json", {"full_object_tree_and_context_equal": exact})
            if not exact:
                raise RuntimeError("Root reproduction gate failed")
            scores = [b["utility_score"] for b in branches]
            root_summary = {"variation": variation, "goal": goal,
                            "root_reproduced": exact, "candidate_count": len(branches),
                            "scores": scores, "candidates": spec["candidates"],
                            "q_has_success_and_lower_utility": max(scores) == 100 and min(scores) < 100,
                            "prefix_controls": len(spec["prefix"]),
                            "first_step_forecast_unique_count": len({b["first_step"]["observation"] for b in branches})}
            summary["roots"].append(root_summary)
            save(output / "prepare_summary.json", summary)
        a, b = [json.loads((output / f"variation_{v}" / "root.json").read_text())
                for v in cfg["variations"]]
        summary["cross_root_comparison"] = {
            "same_goal": a["goal"] == b["goal"],
            "same_actor_context": (a["initial_observation"], a["history"]) == (b["initial_observation"], b["history"]),
            "same_full_object_tree": a["object_tree"] == b["object_tree"],
            "interpretation": "Two fixed development variations, not independent task families."}
        summary["qualified"] = all(r["q_has_success_and_lower_utility"] for r in summary["roots"])
        summary["status"] = "qualified" if summary["qualified"] else "utility_gate_failed"
    except Exception as exc:
        summary.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        if env is not None:
            env.close()
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["scienceworld_version"] = importlib.metadata.version("scienceworld")
        if budget:
            summary["environment_controls"] = {
                "explicit_including_initial_look": budget.explicit_steps,
                "native_gold_generation_action_history_length": budget.gold_generator_actions,
                "total": budget.total,
                "note": "Native gold generation counted by its returned executed action history; no model calls."}
        save(output / "prepare_summary.json", summary)
    print(json.dumps({k: summary.get(k) for k in ("status", "qualified", "elapsed_seconds", "environment_controls", "error")}), flush=True)


def execute(cfg, output):
    """Explicit second phase only; no environment or API call in --prepare."""
    prepared = json.loads((output / "prepare_summary.json").read_text())
    if not prepared.get("qualified") or prepared["status"] != "qualified":
        raise RuntimeError("CPU preparation did not qualify")
    if cfg != json.loads((output / "protocol.json").read_text()):
        raise RuntimeError("Frozen protocol changed after preparation")
    run_dir = output / "actor_execution"
    run_dir.mkdir(exist_ok=False)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "vendor/WorldEvolver/src"))
    from wm_security.client import Client
    from agents.wm_react_agent import WMReactAgent

    class SingleEndpointClient(Client):
        def __init__(self):
            super().__init__(cfg["model"], run_dir, max_calls=cfg["api_attempt_cap"],
                             timeout=cfg["request_timeout_seconds"], retry_truncated=False)
            if self.config.get("provider") != "google_gemini":
                raise RuntimeError("Existing profile is not official Google Gemini")
            if urlsplit(self.config.get("base_url", "")).hostname != "generativelanguage.googleapis.com":
                raise RuntimeError("Existing primary endpoint is not the frozen official host")
            self.config = {k: v for k, v in self.config.items()
                           if not k.startswith(("preferred_", "fallback_", "backup_"))}

        def one(self, system, user, tag):
            if self.calls >= cfg["api_attempt_cap"]:
                raise RuntimeError("Global eight-attempt budget exhausted")
            # The shared client can retry truncations; a per-invocation one-attempt
            # ceiling makes those branches ineligible without editing the client.
            self.max_calls = self.calls + 1
            result = self.completion([{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                                     max_tokens=cfg["max_output_tokens"], temperature=0, tag=tag)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError("Non-stop model completion; no retry permitted")
            return choice["message"]["content"]

    class OneGenerationLLM:
        context_length = 32768
        max_tokens = cfg["max_output_tokens"]
        def __init__(self, client):
            self.client, self.tag, self.used = client, "", False
        def num_tokens_from_messages(self, messages):
            return sum(len(m["content"].encode("utf-8")) + 8 for m in messages)
        def generate(self, system, user):
            if self.used:
                raise RuntimeError("Native actor requested another generation; one-call cap")
            self.used = True
            return True, self.client.one(system, user, self.tag)

    start = time.monotonic()
    result = {"status": "running", "roots": [], "model_api_attempts": 0,
              "scope": cfg["primary_estimand"], "new_environment_controls": 0}
    client = None
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(cfg["api_wall_seconds"])
    try:
        client = SingleEndpointClient()
        llm = OneGenerationLLM(client)
        formatter = original_foresight_formatter()
        for variation in cfg["variations"]:
            directory = output / f"variation_{variation}"
            spec = json.loads((directory / "root_spec.json").read_text())
            root = json.loads((directory / "root.json").read_text())
            branches = json.loads((directory / "branches.json").read_text())
            branch_map = {b["action"]: b for b in branches}
            actor = WMReactAgent(llm, need_goal=True, init_prompt_path=str(
                ROOT / "vendor/WorldEvolver/src/prompts/ReactAgent/scienceworld_react.json"))
            actor.instruction += "\nFor this controlled decision, select exactly one of these actions, verbatim:\n" + "\n".join(spec["candidates"])
            actor.reset(root["goal"], root["initial_observation"])
            for record in root["history"]:
                actor.update(record["action"], record["observation"])
            snapshot = {k: copy.deepcopy(getattr(actor, k)) for k in ("memory", "think_count", "force_action")}
            def choose(tag, foresights):
                for key, value in snapshot.items():
                    setattr(actor, key, copy.deepcopy(value))
                for action in foresights:
                    actor.add_foresight(formatter(action, branch_map[action]["first_step"]["observation"]))
                llm.tag, llm.used = f"variation_{variation}/{tag}", False
                ok, action = actor.run()
                if not ok or action not in spec["candidates"]:
                    raise RuntimeError("One-call actor output is not an exact candidate")
                return {"action": action, "utility_score": branch_map[action]["utility_score"],
                        "success": branch_map[action]["success"],
                        "utility_source": f"variation_{variation}/branches.json: actual environment branch"}
            row = {"variation": variation, "draft": choose("draft", []), "arms": {}}
            result["roots"].append(row)
            save(run_dir / "summary.json", result)
            for arm in cfg["arms"]:
                shown = [] if arm == "sham" else ([row["draft"]["action"]]
                        if arm == "draft_only_truth" else spec["candidates"])
                row["arms"][arm] = choose(arm, shown)
                save(run_dir / "summary.json", result)
            scores = {name: data["utility_score"] for name, data in row["arms"].items()}
            row["candidate_signal"] = (scores["draft_only_truth"] < min(row["draft"]["utility_score"], scores["sham"])
                                       and scores["all_candidate_truth"] >= scores["sham"])
        result["status"] = "completed"
    except Exception as exc:
        # Client sanitizes transport errors before raising; private config is
        # never serialized. Preserve the failed attempt and stop this experiment.
        result.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        result["model_api_attempts"] = client.calls if client else 0
        result["elapsed_seconds"] = time.monotonic() - start
        save(run_dir / "summary.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "model_api_attempts", "elapsed_seconds", "error")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--prepare", action="store_true")
    phase.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = ROOT / cfg["output"]
    (prepare if args.prepare else execute)(cfg, output)


if __name__ == "__main__":
    main()
