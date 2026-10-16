"""Run a bounded upstream discovery agent and save its proposed law for review.

Uses the native no-mid-round-fit option. Generated code is saved here and can be
reviewed before the separate native evaluator executes it. Only transport and
durable logging are adapted; public prompts and the discovery loop are upstream.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/DiscoverPhysics"
sys.path.insert(0, str(VENDOR / "ScienceAgent"))


def main(args):
    import requests
    from scienceagent import llm_client
    from scienceagent.agent import DiscoveryAgent
    from scienceagent.worlds import get_world

    out = Path(args.output).resolve()
    out.mkdir(exist_ok=False)
    save = lambda name, data: (out / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    now = lambda: datetime.now(timezone.utc).isoformat()
    cfg = json.loads(Path("/home/xhj/.config/wm-security/api.json").read_text())
    endpoint = cfg["fallback_url"].rstrip("/")
    key = cfg["fallback_api_key"]

    def redact(text):
        for field in ["api_key", "fallback_api_key"]:
            if cfg.get(field):
                text = text.replace(cfg[field], "[REDACTED]")
        return text

    calls = 0
    records = []
    started = time.monotonic()

    def complete(model, messages, system=None, max_tokens=8192):
        nonlocal calls
        if calls >= args.rounds + 1:
            raise RuntimeError("Bounded native-pilot model-call budget reached")
        body = dict(model=model, messages=([{"role": "system", "content": system}]
                                          if system else []) + messages,
                    max_tokens=max_tokens, stream=False,
                    thinking={"type": "disabled"}, enable_thinking=False)
        calls += 1
        rec = dict(time=time.time(), endpoint=endpoint, request=body,
                   tag=f"native_discovery/{args.world}/call_{calls}",
                   timeout_s=args.timeout)
        save(f"request_{calls:02d}.json", rec)
        save("status.json", dict(status="api_running", calls_started=calls,
                                 native_experiments=len(records), model=model,
                                 primary_called=False, last_progress_at_utc=now()))
        print(json.dumps(dict(event="api_started", model=model, call=calls,
                              prompt_chars=sum(len(x["content"]) for x in body["messages"]))), flush=True)
        t = time.monotonic()
        try:
            response = requests.post(endpoint + "/chat/completions",
                                     headers={"Authorization": "Bearer " + key},
                                     json=body, timeout=(10, args.timeout))
            rec["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            rec["response"] = data
            content = data["choices"][0]["message"].get("content")
            if not content:
                raise RuntimeError("No visible assistant content")
            print(json.dumps(dict(event="api_completed", call=calls,
                                  elapsed_seconds=time.monotonic()-t)), flush=True)
            return content
        except Exception as exc:
            rec["error"] = redact(str(exc))
            print(json.dumps(dict(event="api_failed", model=model,
                                  call=calls, reason=rec["error"],
                                  primary_called=False)), flush=True)
            raise
        finally:
            rec["latency_s"] = time.monotonic() - t
            with (out / "calls.jsonl").open("a") as f:
                f.write(redact(json.dumps(rec, ensure_ascii=False)) + "\n")

    # Preserve upstream prompts, parsing, feedback and experiment execution.
    llm_client.complete = complete
    world = get_world(args.world, engine="nbody", noise_std=0.0)
    executor = world["executor"]
    original_run = executor.run

    def run_experiments(experiments):
        results = []
        for request in experiments:
            save("pending_experiment.json", request)
            response = original_run([request])[0]
            results.append(response)
            records.append(dict(request=request, response=response))
            save("experiments.json", records)
            print(json.dumps(dict(event="native_experiment_completed",
                                  native_experiments=len(records))), flush=True)
        return results

    executor.run = run_experiments
    os.chdir(VENDOR)
    agent = DiscoveryAgent(
        model=args.model, executor=executor, mission=world["mission"],
        max_tokens=8192, verbose=False, max_rounds=args.rounds, min_rounds=2,
        system_prompt_path="PhysicsSchool/prompts/_template_no_mse.md",
        instructions_path=world["instructions"], law_stub=world["law_stub"],
        experiment_format=world["experiment_format"], no_mse=True)
    (out / "system_prompt.txt").write_text(agent._system)
    save("plan.json", dict(
        world=args.world, model=args.model, max_rounds=args.rounds,
        min_rounds=2, max_model_calls=args.rounds+1, max_output_tokens=8192,
        engine="nbody", noise_std=0.0, primary_called=False,
        native_no_mse=True, end_of_mission_fitting="Deferred until law review",
        pin=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=VENDOR, text=True).strip(),
        learner_inputs="Native public system prompt, mission, own experiment responses.",
        hidden_laws_supplied=False,
        scope="One bounded native capability pilot; no method comparison or full benchmark claim."))
    try:
        law = agent.run()
        if law:
            (out / "proposed_law.py").write_text(law + "\n")
        (out / "explanation.txt").write_text(agent.discovered_explanation or "")
        result = dict(status="law_proposed" if law else "no_final_law",
                      law_evaluated=False)
    except Exception as exc:
        result = dict(status="failed", error=redact(str(exc)), law_evaluated=False)
    save("conversation.json", agent.conversation_log)
    result.update(model=args.model, world=args.world, run=str(out.relative_to(ROOT)),
                  api_requests=calls, native_experiments=len(records), gpu_work=False,
                  primary_called=False, elapsed_seconds=time.monotonic()-started,
                  finished_at_utc=now())
    save("status.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="extra_dimensions")
    p.add_argument("--model", default="glm-5.3")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--output", required=True)
    main(p.parse_args())
