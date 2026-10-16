"""Run the unmodified DiscoveryWorld Hypothesizer through our logged client.

This runs short capability and source-curriculum experiments. Hidden scorecards
are recorded for evaluation only. No simulator internals enter model prompts.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from wm_security.client import Client


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/DiscoveryWorld"


class ReplayBoundary(Exception):
    """Dry replay reached a request that was not completed in the source run."""


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with path.open("a") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")


def native_module():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(VENDOR))
    spec = importlib.util.spec_from_file_location(
        "discoveryworld_hypothesizer_native", VENDOR / "agents/HypothesizerAgent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_score(scorecard):
    return [{k: task[k] for k in ("taskName", "score", "maxScore",
                                 "scoreNormalized", "completed",
                                 "completedSuccessfully") if k in task}
            for task in scorecard]


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "knowledge").mkdir()
    native_log = (out / "native_stdout.log").open("w", buffering=1)
    action_script = json.loads(Path(args.action_script).read_text()) if args.action_script else None
    replay_path = Path(args.replay_from).resolve() if args.replay_from else None
    replay_calls = []
    replay_cache = {}
    replay_chain = []
    if replay_path:
        prior_plan = json.loads((replay_path / "plan.json").read_text())
        for key, value in dict(scenario=args.scenario, difficulty=args.difficulty,
                               seed=args.seed, model=args.model,
                               max_steps=args.steps).items():
            assert prior_plan[key] == value, f"Replay configuration changed: {key}"
        assert prior_plan.get("clarify_device_talk", False) == args.clarify_device_talk
        assert prior_plan["generation"].get("reasoning_effort", "none") == args.reasoning_effort
        assert prior_plan["generation"].get("retry_truncated", False) == args.retry_truncated
        prior_script_path = replay_path / "source_actions.json"
        prior_script = json.loads(prior_script_path.read_text()) if prior_script_path.exists() else None
        assert action_script == prior_script, "Replay action curriculum changed"
        # A continuation logs only new requests. Follow its ancestors so a
        # second interruption still replays the entire original trajectory.
        ancestor = replay_path
        while ancestor is not None:
            assert ancestor not in replay_chain, "Cyclic replay ancestry"
            replay_chain.append(ancestor)
            ancestor_plan = json.loads((ancestor / "plan.json").read_text())
            parent = ancestor_plan.get("replay_from")
            ancestor = Path(parent).resolve() if parent else None
        replay_chain.reverse()
        for prior_run in replay_chain:
            prior_calls = prior_run / "calls.jsonl"
            if prior_calls.exists():
                replay_calls.extend(json.loads(line) for line in prior_calls.read_text().splitlines())
        for call in replay_calls:
            choices = call.get("response", {}).get("choices", [])
            content = choices[0].get("message", {}).get("content") if choices else None
            if content and not call.get("error"):
                assert call["tag"] not in replay_cache, "Ambiguous successful completion history"
                replay_cache[call["tag"]] = call
    if action_script is not None:
        assert len(action_script) == args.steps
        save(out / "source_actions.json", action_script)
    revision = subprocess.check_output(
        ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    agent_path = VENDOR / "agents/HypothesizerAgent.py"
    save(out / "plan.json", dict(
        scenario=args.scenario, difficulty=args.difficulty, seed=args.seed,
        model=args.model, max_steps=args.steps, include_images=False,
        upstream="https://github.com/allenai/discoveryworld", commit=revision,
        native_agent_sha256=hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        agent="Unmodified GPT4HypothesizerOneStep and consolidateKnowledgeStep functions",
        substitutions=["Official API model through credential-isolated logged Client",
                       "Text-only setting supported by the native agent",
                       "Headless SDL; run-specific frame directory",
                       "Stop on fresh oracle completion instead of one-observation-late cached completion"],
        generation=dict(temperature=0.1, thinking="disabled" if args.reasoning_effort == "none" else "enabled",
                        reasoning_effort=args.reasoning_effort,
                        retry_truncated=args.retry_truncated,
                        temperature_effective=args.reasoning_effort == "none",
                        native_requested_action_tokens=800,
                        native_requested_knowledge_tokens=3000,
                        client_effective_min_tokens=4096 if args.reasoning_effort == "none" else 8192),
        consolidation="Native schedule: after zero-based step i when i > 0 and i % 10 == 0",
        observation_timing="Native reflection uses the pre-action observation and previous action outcomes (one-step lag preserved).",
        initial_knowledge=str(Path(args.knowledge).resolve()) if args.knowledge else None,
        action_selection="Fixed real-action source curriculum" if action_script is not None else "Native autonomous actor",
        clarify_device_talk=args.clarify_device_talk,
        http_timeout_s=args.api_timeout,
        replay_from=str(replay_path) if replay_path else None,
        replay_ancestors=[str(path) for path in replay_chain],
        replay_policy="Reuse exact previously returned completions only after matching full messages and stage; replay real environment actions from the same seed. Old API requests remain charged only to the original run.",
        scope="Development task or scripted-source experiment; condition-specific scope recorded in the experiment configuration. No cross-seed knowledge transfer.",
        oracle_policy="Scorecards saved separately; never supplied to acting, reflection or consolidation calls.",
        packages={name: importlib.metadata.version(name) for name in
                  ["pygame", "numpy", "openai", "tiktoken", "pathfinding"]}))

    with redirect_stdout(native_log):
        native = native_module()
        api = native.DiscoveryWorldAPI(threadID=args.seed + 1)
        api.FRAME_DIR = str(out / "frames")
        assert api.loadScenario(args.scenario, args.difficulty,
                                randomSeed=args.seed, numUserAgents=1)
        initial_observation = api.getAgentObservation(agentIdx=0)
    save(out / "initial_observation.json", {k: v for k, v in initial_observation.items()
                                            if k != "vision"})
    save(out / "oracle_initial.json", api.getTaskScorecard())
    if args.clarify_device_talk:
        native_action_descriptions = api.listKnownActions

        def action_descriptions(limited=False):
            descriptions = deepcopy(native_action_descriptions(limited=limited))
            descriptions["TALK"]["desc"] = (
                "Talk to an agent, or open the controls of a device listed as an agent "
                "in nearbyAgents. Device controls use dialogue options. arg1 is its UUID.")
            return descriptions

        api.listKnownActions = action_descriptions
    save(out / "actions_schema.json", api.listKnownActions(limited=False))
    client = Client(args.model, out, timeout=args.api_timeout, max_calls=args.max_calls - len(replay_calls),
                    thinking=args.reasoning_effort != "none",
                    reasoning_effort=args.reasoning_effort, retry_truncated=args.retry_truncated)
    context = dict(step=0, phase="step")
    cached_completions = []

    def completion_shim(client, promptStr, promptImages, model=None,
                        prevImage=None, temperature=0.0, maxTokens=800,
                        jsonResponse=False):
        assert not promptImages and prevImage is None, "Text-only pilot"
        phase = ("action" if maxTokens == 800 else "reflection")
        if context["phase"] == "consolidation":
            phase = "consolidation"
        if phase == "action" and action_script is not None:
            # Only action/argument fields are researcher selected. The learner
            # receives real simulator feedback; no domain claims are supplied.
            action = deepcopy(action_script[context["step"]])
            assert set(action) <= {"action", "arg1", "arg2", "chosen_dialog_option_int"}
            answer = json.dumps(dict(action, explanation="Execute the next scheduled environment action.",
                                     memory="", running_hypotheses=[]))
        else:
            messages = [{"role": "user", "content": promptStr}]
            tag = f"step{context['step']:03d}/{phase}"
            cached = replay_cache.get(tag)
            if cached:
                assert cached["request"]["messages"] == messages, f"Replay prompt drift at {tag}"
                answer = cached["response"]["choices"][0]["message"]["content"]
                cached_completions.append(dict(tag=tag, original_id=cached["id"]))
                append(out / "replayed_completions.jsonl", cached_completions[-1])
            elif args.replay_only:
                raise ReplayBoundary(tag)
            else:
                answer = client.chat(messages, max_tokens=maxTokens, temperature=temperature,
                                     tag=tag, json_mode=jsonResponse)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer))])

    native.OpenAIGetCompletion = completion_shim
    native.OPENAI_MODEL_TO_USE = args.model
    history = [{"action": "This is the first action",
                "explanation": "This is the first explanation",
                "memory": "This is the first memory", "running_hypotheses": []}]
    knowledge = (json.loads(Path(args.knowledge).read_text()) if args.knowledge
                 else native.mkInitialHypotheses())
    assert isinstance(knowledge.get("scientific_knowledge"), list)
    if replay_path:
        assert knowledge == json.loads((replay_path / "knowledge/initial.json").read_text()), "Initial knowledge changed"
    save(out / "knowledge/initial.json", knowledge)
    previous_observation = None
    scorecard = api.getTaskScorecard()
    completed_steps = 0
    started = time.monotonic()
    summary = dict(status="running", model=args.model, scenario=args.scenario,
                   difficulty=args.difficulty, seed=args.seed, steps=0)
    save(out / "summary.json", summary)

    # Record actual mutation boundaries even if a later reflection request fails.
    real_action = api.performAgentAction
    real_tick = api.tick

    def action_with_record(agentIdx, actionJSON):
        result = real_action(agentIdx=agentIdx, actionJSON=actionJSON)
        append(out / "environment_events.jsonl", dict(
            step=context["step"], kind="action", action=actionJSON, result=result))
        return result

    def tick_with_record():
        result = real_tick()
        append(out / "environment_events.jsonl", dict(
            step=context["step"], kind="tick", world_step=api.world.getStepCounter()))
        return result

    api.performAgentAction = action_with_record
    api.tick = tick_with_record
    print(json.dumps(dict(event="loaded", scenario=args.scenario,
                          difficulty=args.difficulty, seed=args.seed,
                          model=args.model, commit=revision)), flush=True)
    try:
        for i in range(args.steps):
            context.update(step=i, phase="step")
            with redirect_stdout(native_log):
                packed = native.GPT4HypothesizerOneStep(
                    api, client, history, previous_observation,
                    knowledge, includeImages=False)
            action = packed["nextAction"]
            if "errors" in packed["actionSuccess"]:
                action["errors"] = packed["actionSuccess"]["errors"]
            history.append(action)
            previous_observation = packed["observation"]
            scorecard = packed.pop("oracle_scorecard")
            append(out / "oracle_scorecards.jsonl", dict(step=i, scorecard=scorecard))
            compact = deepcopy(packed)
            compact["observation"].pop("vision", None)
            append(out / "steps.jsonl", dict(index=i, **compact))
            save(out / f"knowledge/step_{i:03d}_before_consolidation.json", knowledge)
            if i > 0 and i % 10 == 0:
                context["phase"] = "consolidation"
                with redirect_stdout(native_log):
                    knowledge = native.consolidateKnowledgeStep(client, knowledge, stepIdx=i)
                save(out / "consolidation.json", native.CONSOLATATE_TRACKING)
            save(out / f"knowledge/step_{i:03d}.json", knowledge)
            save(out / "action_history.json", history)
            completed_steps = i + 1
            print(json.dumps(dict(step=i + 1,
                                  action={k: action[k] for k in
                                          ["action", "arg1", "arg2", "chosen_dialog_option_int"]
                                          if k in action},
                                  action_success=packed["actionSuccess"],
                                  scorecard=public_score(scorecard),
                                  knowledge_entries=len(knowledge["scientific_knowledge"]),
                                  calls=client.calls)), flush=True)
            if scorecard and all(task["completed"] for task in scorecard):
                break
        summary["status"] = "completed" if all(t["completed"] for t in scorecard) else "budget_exhausted"
    except ReplayBoundary as exc:
        summary.update(status="replay_ready", next_request=str(exc))
    except BaseException as exc:
        summary.update(status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary.update(steps=completed_steps, api_calls=client.calls,
                       replayed_completions=len(cached_completions),
                       prior_api_calls=len(replay_calls),
                       wall_seconds=time.monotonic() - started,
                       scorecard=public_score(api.getTaskScorecard()))
        save(out / "summary.json", summary)
        save(out / "oracle_final.json", api.getTaskScorecard())
        save(out / "knowledge/final.json", knowledge)
        native_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Easy")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-calls", type=int, default=260)
    parser.add_argument("--api-timeout", type=float, default=90,
                        help="Base HTTP wait limit in seconds; does not change model request parameters")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "high", "max"], default="none")
    parser.add_argument("--retry-truncated", action="store_true",
                        help="Apply the bounded larger-budget retry to nonempty token-truncated responses too")
    parser.add_argument("--clarify-device-talk", action="store_true",
                        help="Clarify TALK device affordance using public nearbyAgents metadata; no task answers")
    parser.add_argument("--knowledge", help="Previously learned native scientific_knowledge JSON")
    parser.add_argument("--action-script", help="JSON array of real actions; overrides actor only for source collection")
    parser.add_argument("--replay-from", help="Continue an interrupted run in a fresh output directory using exact cached completions")
    parser.add_argument("--replay-only", action="store_true", help="Stop before any new API request")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
