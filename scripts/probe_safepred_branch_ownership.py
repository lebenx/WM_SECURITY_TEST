#!/usr/bin/env python3
"""Pinned native AST boundary probe; deterministic fixtures, no model or task execution."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "agent/agent.py", "wrapper.py", "models/prompts.py", "utils/state_preprocessor.py",
    "benchmark/wasp/visualwebarena/agent/agent.py",
    "benchmark/wasp/visualwebarena/agent/safepred_wrapper.py", "models/world_model.py",
]
AST_SELECTION = {
    "utils/state_preprocessor.py": {"StatePreprocessor": None},
    "models/prompts.py": {"PromptTemplates": ["world_model_state_delta", "_format_compact_state", "_format_state_with_full_tree"]},
    "models/world_model.py": {"LLMBasedWorldModel": ["simulate", "_create_prompt"]},
    "agent/agent.py": {
        "SearchPath": None,
        "SafeAgent": ["_preprocess_state", "_synthesize_compact_state_from_prediction", "_tree_search_planning",
                      "_extract_metadata_fields", "_select_best_path", "_expand_paths_at_depth", "get_safe_action"],
    },
    "wrapper.py": {"SafePredWrapper": ["evaluate_action_risk", "evaluate_action_risk_with_plan", "_determine_plan_update",
                                         "_format_state_for_action_prompt", "_get_default_action_prompt_template"]},
}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class HistoryHarness:
    """Explicit isolated history service, not an upstream storage implementation."""
    def __init__(self):
        self.messages = [{"role": "user", "message": "Complete the fixed harmless UI task."}]
        self.saved = []
    def set_task(self, **kwargs):
        self.task = kwargs
    def get_conversation_history(self):
        return self.messages
    def get_total_executed_steps(self):
        return 0
    def add_executed_response(self, **kwargs):
        self.saved.append(deepcopy(kwargs))  # Native label does not imply execution in this probe.


class AdapterHarness:
    """Trajectory-list to raw-observation projection; action tokens are fixed strings."""
    def state_to_safepred(self, raw_state, intent, metadata):
        raw = raw_state[-1]["observation"] if isinstance(raw_state, list) else raw_state
        out = deepcopy(raw)
        out["goal"] = intent
        return out
    def action_to_safepred(self, action):
        assert isinstance(action, str)
        return action
    def action_from_safepred(self, action):
        return action


def prohibited(*args, **kwargs):
    raise RuntimeError("Unexpected external or unqualified harness path")


def extract(out, sources, manifest):
    """Compile untouched class/method AST nodes in isolated dependency namespaces."""
    package = ModuleType("safepred_probe")
    package.__path__ = []
    sys.modules[package.__name__] = package
    models = ModuleType("safepred_probe.models")
    models.__path__ = []
    sys.modules[models.__name__] = models
    common = {"logger": SilentLogger(), "re": re, "ET": ET, "dataclass": dataclass,
              "json": json, "datetime": datetime, "format_policies_with_references": prohibited}
    def validate_dict(value, *args, **kwargs):
        if not isinstance(value, dict):
            raise TypeError("Native result is not a dict")
    common["validate_dict"] = validate_dict
    classes = {}
    for path, selected in AST_SELECTION.items():
        tree = ast.parse(sources[path], filename=str(out / "source" / path))
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        for name, methods in selected.items():
            original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            node = deepcopy(original)
            if methods is not None:
                node.body = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
                assert {n.name for n in node.body} == set(methods)
                # Constructors/base dependencies are bypassed, never silently imported.
                node.bases = []
                node.keywords = []
                node.decorator_list = []
            for method in [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                old = next(n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method.name)
                assert ast.dump(method, include_attributes=True) == ast.dump(old, include_attributes=True)
                manifest.append({"source": path, "class": name, "method": method.name,
                                 "line": method.lineno, "end_line": method.end_lineno,
                                 "unchanged_method_ast": True,
                                 "ast_sha256": digest(ast.dump(method, include_attributes=False).encode())})
            body.append(node)
        name = "safepred_probe.models.prompts" if path == "models/prompts.py" else "safepred_probe.extracted_" + str(len(classes))
        module = ModuleType(name)
        module.__package__ = "safepred_probe.models" if path == "models/world_model.py" else "safepred_probe"
        module.__dict__.update(common)
        module.__dict__.update(classes)
        sys.modules[name] = module
        code = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(code, str(out / "source" / path), "exec"), module.__dict__)
        for key in selected:
            classes[key] = module.__dict__[key]
    return classes


def raw_fixture(tree):
    return {"axtree_txt": tree, "goal": "Complete the fixed harmless UI task.", "policies": [],
            "current_action_reasoning": "Consider the fixed sibling candidate."}


def bare_agent(classes):
    agent = classes["SafeAgent"].__new__(classes["SafeAgent"])
    agent.state_preprocessor = classes["StatePreprocessor"](benchmark="visualwebarena")
    agent.conversation_history_manager = HistoryHarness()
    agent.policies = []
    return agent


def prompt_pair(classes, wrapper, state):
    wm = classes["PromptTemplates"].world_model_state_delta(state, "fixed_sibling", policies=[], show_policy_references=False)
    action = wrapper._get_default_action_prompt_template().format(
        state=wrapper._format_state_for_action_prompt(state), goal=state["goal"], num_samples=1,
        risk_guidance="", policies="")
    return wm, action


def state_alias_probe(classes, out, fixtures):
    rows = []
    wrapper = classes["SafePredWrapper"].__new__(classes["SafePredWrapper"])
    for raw_name, raw in fixtures["raw_states"].items():
        child_reference = None
        for arm in ["native_shallow", "deepcopy_input", "copy_history_only"]:
            agent = bare_agent(classes)
            parent = agent._preprocess_state(deepcopy(raw))
            assert "axtree_txt" in parent
            assert bool(parent["key_elements"]) == bool(raw["axtree_txt"])
            before = deepcopy(parent)
            before_prompts = prompt_pair(classes, wrapper, parent)
            if arm == "deepcopy_input":
                synthesis_input = deepcopy(parent)
            elif arm == "copy_history_only":
                synthesis_input = parent.copy()
                synthesis_input["conversation_history"] = deepcopy(parent["conversation_history"])
            else:
                synthesis_input = parent
            child = agent._synthesize_compact_state_from_prediction(synthesis_input, "synthetic branch state", deepcopy(fixtures["alias_delta"]))
            after_prompts = prompt_pair(classes, wrapper, parent)
            if child_reference is None:
                child_reference = deepcopy(child)
            row = {
                "raw_input": raw_name, "arm": arm,
                "parent_mutated": parent != before,
                "parent_key_elements_mutated": parent["key_elements"] != before["key_elements"],
                "parent_child_key_elements_same_object": parent["key_elements"] is child["key_elements"],
                "child_content_matches_native": child == child_reference,
                "protected_fields_exact": all(parent[k] == before[k] for k in before if k != "key_elements"),
                "world_model_prompt_bytes_equal": before_prompts[0].encode() == after_prompts[0].encode(),
                "default_action_prompt_bytes_equal": before_prompts[1].encode() == after_prompts[1].encode(),
                "parent_before": before, "parent_after": deepcopy(parent), "child": deepcopy(child),
            }
            prefix = out / "prompts" / f"{raw_name}_{arm}"
            prefix.parent.mkdir(exist_ok=True)
            for i, kind in enumerate(["world_model", "action"]):
                for when, pair in [("before", before_prompts), ("after", after_prompts)]:
                    Path(f"{prefix}_{kind}_{when}.txt").write_text(pair[i])
            rows.append(row)
    return rows


def guidance_probe(classes, protocol, fixtures):
    p = protocol["guidance_probe"]
    cases = [("default_singleton_depth1", 1, ["safe_root"], None)]
    cases += [(f"multiroot_depth1_order{i}", 1, order, None) for i, order in enumerate(p["root_orders"])]
    cases += [(f"single_root_depth2_order{i}", 2, ["safe_root"], order) for i, order in enumerate(p["single_root_child_orders"])]
    rows = []
    for name, depth, roots, children in cases:
        events = []
        agent = bare_agent(classes)
        agent.config = SimpleNamespace(tree_search_max_depth=depth, tree_search_n_root=len(roots),
                                       tree_search_n_child=len(children) if children else 1, tree_search_m_child=1,
                                       root_risk_threshold=p["root_and_child_threshold"],
                                       child_risk_threshold=p["root_and_child_threshold"], enable_pruning=True,
                                       show_policy_references=False)
        agent._save_filtered_action = lambda **kwargs: events.append({"kind": "isolated_filtered_storage", "args": deepcopy(kwargs)})
        wm = classes["LLMBasedWorldModel"].__new__(classes["LLMBasedWorldModel"])
        wm._model_loaded = True
        wm._use_api = True  # Native dispatch reaches the explicit fixture endpoint below, never an API.
        wm.api_key, wm.api_url = "fixture-not-a-credential", "fixture://no-network"
        wm.prediction_steps, wm.use_state_delta = 1, True
        wm.use_few_shot, wm.few_shot_examples = False, []
        wm.show_policy_references = False
        wm.enable_short_term_prediction = wm.enable_long_term_prediction = True
        wm._to_string = str  # Fixtures are already string action tokens; identity on this domain.
        def response(prompt, state, action):
            wm._last_predicted_delta = deepcopy(fixtures["deltas"][action])
            wm.last_action = action
            events.append({"kind": "synthetic_response", "action": action,
                           "delta": deepcopy(wm._last_predicted_delta), "prompt": prompt,
                           "state": deepcopy(state)})
            return "Synthetic fixture state; not an environment observation."
        wm._simulate_api = response
        agent.world_model = wm
        native_result = []
        original_get = agent.get_safe_action
        def observed_get(*args, **kwargs):
            result = original_get(*args, **kwargs)
            native_result.append(deepcopy(result))
            return result
        agent.get_safe_action = observed_get
        wrapper = classes["SafePredWrapper"].__new__(classes["SafePredWrapper"])
        wrapper.safe_agent, wrapper.config = agent, agent.config
        wrapper.adapter, wrapper.policy_manager = AdapterHarness(), None
        wrapper.use_planning, wrapper.web_agent_llm_config = True, None
        monitor_calls = []
        def monitor(**kwargs):
            monitor_calls.append(deepcopy(kwargs))
            return {"optimization_guidance": ""}
        wrapper.plan_monitor = SimpleNamespace(monitor_with_plan=monitor)
        def create_generator(**kwargs):
            def generate(state, risk_guidance, num_samples, simulated_path):
                events.append({"kind": "synthetic_child_proposals", "state": deepcopy(state),
                               "simulated_path": simulated_path, "num_samples": num_samples})
                assert num_samples == len(children)
                return list(children)
            return generate
        wrapper._create_default_action_generator = create_generator
        native_wrapper_result = []
        original_evaluate = wrapper.evaluate_action_risk
        def observed_evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            native_wrapper_result.append(deepcopy(result))
            return result
        wrapper.evaluate_action_risk = observed_evaluate
        # Match WASP's trajectory-list shape; do not inject plan_text into its observation.
        trajectory = [{"observation": deepcopy(fixtures["raw_states"]["nonempty_tree"])}]
        call = dict(state=trajectory, action="safe_root", plan_text="The same fixed plan in every case.",
                    intent=trajectory[-1]["observation"]["goal"], metadata={"task_id": "synthetic_fixed_root"})
        if len(roots) > 1:
            call["candidate_actions"] = list(roots)  # Public API experiment, not the WASP call signature.
        result = wrapper.evaluate_action_risk_with_plan(**call)
        assert len(native_result) == len(native_wrapper_result) == 1
        rows.append({
            "case": name, "configured_depth": depth, "with_plan_depth_argument": "native_default_1",
            "root_actions": roots, "child_actions": children,
            "input_boundary": "WASP-shaped trajectory list through explicit adapter harness",
            "reachability": "supported nondefault public multicandidate API" if len(roots) > 1 else
                            ("default singleton" if depth == 1 else "single-root WASP API; nondefault depth2 and child count3"),
            "selected_action": result["selected_action"], "native_selected_result": native_result[0],
            "selected_root_evaluation": deepcopy(agent._last_action_evaluation),
            "wrapper_before_plan_result": native_wrapper_result[0], "with_plan_result": result,
            "last_predicted_action": wm.last_action, "last_predicted_delta": deepcopy(wm._last_predicted_delta),
            "plan_monitor_calls": len(monitor_calls),
            "replay_simulate_calls": len([e for e in events if e["kind"] == "synthetic_response"]),
            "history_service_writes": deepcopy(agent.conversation_history_manager.saved), "events": events,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/safepred_branch_ownership_probe_20261015.json")
    parser.add_argument("--out", default="runs/safepred_branch_ownership_probe_20261015")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = json.loads((ROOT / args.config).read_text())
    shutil.copy2(ROOT / args.config, out / "frozen_protocol.json")
    shutil.copy2(__file__, out / "launched_script.py")
    summary = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
               "interpreter": sys.executable, "python": sys.version, "upstream": protocol["upstream"],
               "activity": {"model_calls": 0, "model_api_calls": 0, "environment_steps": 0,
                            "checkpoint_loads": 0, "training_updates": 0, "gpu_use": False,
                            "source_git_show_calls": 0, "runner_network_requests": 0,
                            "replay_simulate_calls": 0}, "source_manifest": [], "ast_manifest": []}
    save(out / "summary.json", summary)
    counts = Counter()
    trace = {}
    try:
        sources = {}
        git_env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
        for path in SOURCE_PATHS:
            summary["activity"]["source_git_show_calls"] += 1
            completed = subprocess.run(["git", "-c", "gc.auto=0", "-c", "protocol.allow=never", "-C",
                                        str(ROOT / protocol["upstream"]["local_path"]), "show",
                                        f"{protocol['upstream']['commit']}:{path}"], env=git_env,
                                       capture_output=True, check=True, timeout=30)
            target = out / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(completed.stdout)
            sources[path] = completed.stdout.decode()
            summary["source_manifest"].append({"path": path, "bytes": len(completed.stdout),
                "sha256": digest(completed.stdout), "method": "git show; lazy fetch disabled; network protocols denied",
                "stderr": completed.stderr.decode()})
        classes = extract(out, sources, summary["ast_manifest"])
        p = protocol["guidance_probe"]
        fixtures = {"raw_states": {"nonempty_tree": raw_fixture("[1] [button] [Existing control]\n"),
                                    "empty_tree_key_retained": raw_fixture("")},
                    "alias_delta": {"element_changes": {"new_elements": [{"label": "HYPOTHETICAL_ONLY_SENTINEL",
                                              "type": "button", "bid": "probe_only"}], "removed_elements": []}},
                    "deltas": {action: {"risk_score": risk, "risk_explanation": "Fixed synthetic risk for " + action,
                            "violated_policy_ids": ["SYNTHETIC_POLICY"] if action in p["guidance"] else [],
                            "optimization_guidance": p["guidance"].get(action),
                            "element_changes": {"new_elements": [], "removed_elements": []},
                            "semantic_delta": "Fixed synthetic delta for " + action}
                            for action, risk in p["fixed_risks"].items()}}
        save(out / "fixtures_frozen_before_calls.json", fixtures)
        source_prefix = str(out / "source") + os.sep
        def profiler(frame, event, arg):
            if event == "call" and frame.f_code.co_filename.startswith(source_prefix):
                filename = frame.f_code.co_filename[len(source_prefix):]
                counts[f"{filename}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}"] += 1
        sys.setprofile(profiler)
        trace["state_alias"] = state_alias_probe(classes, out, fixtures)
        trace["guidance"] = guidance_probe(classes, protocol, fixtures)
        sys.setprofile(None)
        # Evaluate the frozen operator questions; no extra candidate/fixture search.
        state_rows = trace["state_alias"]
        guidance_rows = trace["guidance"]
        summary["state_alias_table"] = [{k: v for k, v in r.items() if k not in {"parent_before", "parent_after", "child"}} for r in state_rows]
        summary["guidance_table"] = [{"case": r["case"], "selected_action": r["selected_action"],
            "selected_path": r["native_selected_result"]["path"],
            "selected_root_violated_policy_ids": r["selected_root_evaluation"]["world_model_metadata"]["violated_policy_ids"],
            "last_predicted_action": r["last_predicted_action"],
            "optimization_guidance": r["with_plan_result"]["optimization_guidance"],
            "should_update_plan": r["with_plan_result"]["should_update_plan"],
            "native_should_check_plan": r["native_selected_result"]["should_check_plan"],
            "wrapper_keeps_should_check_plan": "should_check_plan" in r["wrapper_before_plan_result"],
            "wrapper_keeps_path": "path" in r["wrapper_before_plan_result"],
            "plan_monitor_calls": r["plan_monitor_calls"], "replay_simulate_calls": r["replay_simulate_calls"]} for r in guidance_rows]
        summary["checks"] = {
            "all_child_contents_equal_across_copy_arms": all(r["child_content_matches_native"] for r in state_rows),
            "all_protected_fields_exact": all(r["protected_fields_exact"] for r in state_rows),
            "all_default_action_prompts_unchanged": all(r["default_action_prompt_bytes_equal"] for r in state_rows),
            "nonempty_tree_world_model_prompts_unchanged": all(r["world_model_prompt_bytes_equal"] for r in state_rows if r["raw_input"] == "nonempty_tree"),
            "default_singleton_has_no_plan_update": not guidance_rows[0]["with_plan_result"]["should_update_plan"],
            "all_selected_actions_safe_root": all(r["selected_action"] == "safe_root" for r in guidance_rows),
            "no_progress_step_fixture_or_result": all("progress_step" not in r["last_predicted_delta"] and "progress_step" not in r["with_plan_result"] for r in guidance_rows),
        }
        summary["activity"]["replay_simulate_calls"] = sum(r["replay_simulate_calls"] for r in guidance_rows)
        summary["status"] = "completed_cpu_native_operator_boundary_probe"
        summary["interpretation"] = {
            "kind": "Synthetic code/operator evidence only; not an attack or task experiment.",
            "state_alias": "Compare the six rows: deep input copy tests list ownership; full-tree and default-action formatting are native negative controls. Empty-tree fallback is a separate edge case, not a modified factual tree.",
            "guidance": "Nondefault order dependence can be traced to the global last predicted delta; selected root metadata remains separately bound. Risk guidance from rejected hypothetical actions can be intended design, so order dependence alone is not harmful contamination.",
            "consumer": "Pinned WASP source statically routes returned guidance to a plan LLM and current_plan; that LLM and the next action were not executed.",
            "limitations": ["No real forecast, risk correctness, environment observation or actual utility.",
                "Candidate tokens and child proposals are fixed fixtures, not real LLM proposals.",
                "No real benchmark adapter, storage, history manager or full constructors were executed.",
                "WASP-shaped trajectory list is projected by an explicit adapter harness; with_plan does not inject plan_text into the list.",
                "One-root depth2 is supported nondefault; multi-root candidates are only the public API boundary.",
                "No progress_step is inserted: the native prompt explicitly omits it.",
                "No conclusion that all possible nested-state mutations or normal deployed behavior are covered.",
                "Runner source fetch is network-disabled; this does not claim the parent research turn used zero web requests."]}
        summary["harness_substitutions"] = [
            "AST class shells bypass constructors and upstream imports; selected method bodies are unchanged.",
            "simulate and _create_prompt are native; _simulate_api returns fixed parsed delta fixtures, with no API client.",
            "Action-to-string is identity for string fixtures; child action generator returns the prescribed order.",
            "Adapter projects a synthetic trajectory observation into a copied raw observation and maps string actions identically.",
            "History/logging/filtered storage are local harness services; policy manager is None and policy references are disabled.",
            "PlanMonitor is a recording no-guidance fixture; counts disclose whether native routing reaches it.",
            "validate_dict enforces the corresponding type assertion; no external type-checker package is imported.",
            "Transparent recording wrappers retain exact native get_safe_action/evaluate_action_risk return objects."]
        # Only static evidence for the external WASP consumer (not evaluated).
        summary["wasp_static_source_hits"] = {}
        for path in SOURCE_PATHS[4:6]:
            hits = []
            for i, line in enumerate(sources[path].splitlines(), 1):
                if any(s in line for s in ["evaluate_action_risk_with_plan", "should_update_plan", "generate_plan_with_guidance",
                    "self.current_plan = updated_plan", "format_plan_for_prompt(self.current_plan)", "from SafePred", "from safepred"]):
                    hits.append({"line": i, "text": line.strip()})
            summary["wasp_static_source_hits"][path] = hits
    except Exception as exc:
        sys.setprofile(None)
        summary["status"] = "harness_error"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        summary["ast_call_counts"] = dict(sorted(counts.items()))
        summary["elapsed_seconds"] = time.monotonic() - started
        save(out / "trace.json", trace)
        save(out / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                          "activity": summary["activity"], "summary": str(out / "summary.json")}), flush=True)
    if summary["status"] != "completed_cpu_native_operator_boundary_probe":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
