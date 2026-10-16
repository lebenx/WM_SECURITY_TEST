"""Bounded native Alice repair continuation from a released program.

The initial program is author-supplied. The new preservation set is constructed
from correctly predicted training-map solution transitions, in one new root
class; it does not reconstruct the author's online history or class tree.
Only the native manual-data collector is used; no frontier explorer is trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/BabaInWonderland"
sys.path.insert(0, str(UPSTREAM))

from scripts.data_collect.evaluate_offline_dataset import _load_state_archive_inputs
from scripts.data_collect.data_coverage import archive_state_input_to_canonical_text
from src.agents.manual_transition_explorer import ManualTransitionExplorer
from src.data import Transition
from src.discovery.program_pipeline import ProgramDiscoveryPipeline
from src.llm.inference import GenerationConfig, LLMPredictor
from src.program_model.evaluator import ProgramEvaluator


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class LoggedPredictor(LLMPredictor):
    """Log each native HTTP attempt without changing its request or response."""

    def __init__(self, config, output, call_cap, stream=False):
        super().__init__(config)
        self.output = output
        self.call_cap = call_cap
        self.calls = 0
        self.stream = stream

    def _post_json(self, req, timeout_sec, provider_name):
        if self.calls >= self.call_cap:
            raise RuntimeError("Screen HTTP call cap exhausted")
        self.calls += 1
        payload = json.loads(req.data)
        if self.stream:
            payload["stream"] = True
        record = {"id": str(uuid.uuid4()), "tag": "baba_native_repair",
                  "time": time.time(), "endpoint": self.config.openai_base_url,
                  "primary_called": False, "request": payload,
                  "timeout_s": timeout_sec}
        started = time.monotonic()
        print(json.dumps({"api_started": self.calls, "model": self.config.openai_model,
                          "endpoint": self.config.openai_base_url}), flush=True)
        try:
            if self.stream:
                response = self._stream_completion(req.full_url, payload, record, started)
            else:
                response = super()._post_json(req, timeout_sec, provider_name)
            record["response"] = response
            return response
        except Exception as exc:
            record["error"] = str(exc).replace(self.config.openai_api_key, "[REDACTED]")
            raise
        finally:
            record["latency_s"] = time.monotonic() - started
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            text = json.dumps(record, ensure_ascii=False).replace(self.config.openai_api_key, "[REDACTED]")
            with (self.output / "calls.jsonl").open("a") as handle:
                handle.write(text + "\n")
            print(json.dumps({"api_finished": self.calls, "latency_s": record["latency_s"],
                              "error": record.get("error")}), flush=True)

    def _stream_completion(self, url, payload, record, started):
        content, reasoning = [], []
        finish_reason = None
        usage = {}
        with requests.post(url, headers={"Authorization": "Bearer " + self.config.openai_api_key},
                           json=payload, stream=True, timeout=(10, 90)) as response:
            record["http_status"] = response.status_code
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if time.monotonic() - started > 240:
                    raise TimeoutError("Streaming response exceeded 240-second screen window")
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                if not raw:
                    continue
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        record.setdefault("first_content_latency_s", time.monotonic() - started)
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        if not content or finish_reason is None:
            raise RuntimeError("Incomplete streaming completion")
        return {"choices": [{"message": {"content": "".join(content),
                                         "reasoning_content": "".join(reasoning)},
                             "finish_reason": finish_reason}], "usage": usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--max-calls", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stream", action="store_true",
                        help="SSE transport adapter; preserves native prompt and generation parameters.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    probe = ROOT / "runs/baba_program_history_pair_v0"
    split = yaml.safe_load((UPSTREAM / "configs/custom_map_splits_dot_test.yaml").read_text())["original"]
    aliases = yaml.safe_load((UPSTREAM / "configs/experiment_config_online.yaml").read_text())["serialization"]["word_aliases"]
    metadata = json.loads((probe / "wonderland_transitions.json").read_text())
    with np.load(probe / "wonderland_correctness.npz", allow_pickle=False) as payload:
        assert str(payload["versions"][-1]) == "v050"
        mask = payload["correct"][-1]
    program_path = ROOT / "vendor/BabaInWonderland_demo/programs/wonderland/v050.py"
    source = program_path.read_text()
    eval_rows = [json.loads(line) for line in (probe / "programs.jsonl").read_text().splitlines()]
    prior = next(row for row in eval_rows if row["variant"] == "wonderland" and row["version"] == "v050")
    assert prior["program_sha256"] == sha(source)
    transitions = []
    ordered_metadata = []
    for world_index, map_name in enumerate(split["all_train"], start=1):
        raw = _load_state_archive_inputs(UPSTREAM / "test_dataset/solution" / f"{map_name}_states.npz")
        states = [archive_state_input_to_canonical_text(s, word_aliases=aliases) for s in raw]
        row_path = UPSTREAM / "test_dataset/solution" / f"{map_name}_transitions.jsonl"
        for row in map(json.loads, filter(None, row_path.read_text().splitlines())):
            transitions.append(Transition(state=states[row["state_index"]], action=row["action"],
                next_state=states[row["next_state_index"]], reward=row.get("reward",0),
                done=bool(row.get("done",False)), world_index=world_index, map_name=map_name))
            ordered_metadata.append({"map": map_name,"transition_index":row["transition_index"],"action":row["action"]})
    assert ordered_metadata == metadata
    target_index = int(np.flatnonzero(~mask)[0])
    target = transitions[target_index]
    correct = [t for t, c in zip(transitions, mask) if c]
    manual = args.output / "manual"
    manual.mkdir()
    save(manual / "target.json", {
        "previous_state_raw": target.state, "next_state_raw": target.next_state,
        "action": target.action, "reward": target.reward, "done": target.done,
        "env_index": target.world_index, "scenario_type": target.map_name,
        "step_index": metadata[target_index]["transition_index"],
    })
    config = json.loads(Path('/home/xhj/.config/wm-security/api.json').read_text())
    predictor = LoggedPredictor(GenerationConfig(
        openai_base_url=config["fallback_url"],
        openai_api_key=config["fallback_api_key"], openai_model=args.model,
        max_tokens=8192, openai_timeout_sec=120, request_max_retries=0,
    ), args.output, args.max_calls, stream=args.stream)
    native = args.output / "native"
    native.mkdir()
    evaluator = ProgramEvaluator(program_eval_workers=1)
    pipeline = ProgramDiscoveryPipeline(
        explorer=ManualTransitionExplorer(manual, seed=0), llm_predictor=predictor,
        output_dir=str(native), evaluator=evaluator,
        max_fail_count=args.max_calls, max_total_llm_calls=args.max_calls,
        patch_unexpected_error_max_attempts=1, max_collect_transitions_per_verify=256,
        saturation_global_step_window=300000, shuffle_collected_transitions=False,
        regression_group_sample_k=3, regression_witnesses_per_group=1,
        patch_format="function", save_new_transition_images=False, random_seed=0,
    )
    pipeline.current_version = pipeline.repository.save_version(
        source, pipeline.current_version.version_id,
        {"reason":"Research initialization from published Wonderland v050; not a new learned update."})
    pipeline.current_source = source
    pipeline.canonical_d.merge(correct)
    for transition in pipeline.canonical_d.get_all():
        pipeline._add_to_protected(transition, commit_version_override=pipeline.current_version.version_id)
    pipeline._sync_explorer_program_context(emit_log=False)
    initial_class_count = pipeline._current_active_class_count()
    save(args.output / "setup.json", {
        "initial_program": str(program_path.relative_to(ROOT)), "program_sha256": sha(source),
        "script_sha256": sha(Path(__file__).read_text()), "target": metadata[target_index],
        "target_selection": "First incorrect training-map solution transition in manifest and trajectory order.",
        "correct_archived_rows": len(correct), "protected_unique_transitions": len(pipeline.protected_set),
        "initial_class_count": initial_class_count,
        "preservation_initialization": "New root class from all correctly predicted train solution transitions; not the author's original protected history or split tree.",
        "native_components": ["manual collector", "patch prompt and generator", "patcher", "sandbox and evaluator", "acceptance gate", "regression feedback", "class refinement"],
        "model": args.model, "max_http_calls": args.max_calls, "primary_called": False,
        "stream_transport_adapter": args.stream,
        "new_environment_interactions": 0, "gpu_work": False,
        "scope": "Offline repair of one actual prediction failure; no exploration or attack measurement.",
    })
    print(json.dumps({"prepared": True, "protected_unique_transitions": len(pipeline.protected_set),
                      "target": metadata[target_index], "initial_class_count": initial_class_count}), flush=True)
    if args.prepare_only:
        evaluator.close()
        return
    error = None
    try:
        pipeline.run(max_iterations=5)
    except Exception as exc:
        error = str(exc).replace(config["fallback_api_key"], "[REDACTED]")
        print(json.dumps({"error": error}), flush=True)
    final = evaluator.evaluate_source(pipeline.current_source, transitions)
    initial_keys = {pipeline._transition_key(t) for t in correct}
    lost = [r.index for r in final.records if not r.is_correct and pipeline._transition_key(r.transition) in initial_keys]
    summary = {
        "status": "completed" if error is None else "failed", "error": error,
        "model": args.model, "api_requests": predictor.calls, "primary_called": False,
        "initial_correct": int(mask.sum()), "final_correct": final.correct_count,
        "total": len(transitions), "initial_class_count": initial_class_count,
        "final_class_count": pipeline._current_active_class_count(),
        "accepted_new_versions": pipeline.current_version.index - 1,
        "final_program_unchanged": pipeline.current_source == source,
        "protected_losses": lost, "target_solved": final.records[target_index].is_correct,
        "wall_time_s": time.monotonic()-start, "gpu_work": False,
        "new_environment_interactions": 0, "autonomous_task_episodes": 0,
        "termination_reason": pipeline._last_termination_reason,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Single target and model; no attack or causal learning-harm comparison.",
                        "Research-initialized preservation set differs from the published online training history.",
                        "Manual collector does not train or exercise the native frontier explorer."],
    }
    save(args.output / "summary.json", summary)
    evaluator.close()
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
