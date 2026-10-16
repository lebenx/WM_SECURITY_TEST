"""Collect task completion and actual interaction costs for the native pilot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "cold_no_thinking": "discoveryworld_reactor_pro_clean_v0",
    "warm_no_thinking": "discoveryworld_reactor_pro_warm_v0",
    "cold_high_thinking": "discoveryworld_reactor_pro_reasoning_clean_v0",
    "warm_high_thinking": "discoveryworld_reactor_pro_reasoning_warm_v0",
    "cold_flash_high_v1": "discoveryworld_reactor_flash_reasoning_clean_v1",
    "warm_flash_high_v1": "discoveryworld_reactor_flash_reasoning_warm_v1",
    "cold_gateway_flash_v1": "discoveryworld_reactor_glm53flash_clean_v1_resume2",
    "warm_gateway_flash_v1": "discoveryworld_reactor_glm53flash_warm_v1",
}


def jsonl(path):
    if not path.exists():
        return []
    # A running writer may not have finished its last line yet.
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def collect(folder):
    path = ROOT / "runs" / folder
    if not (path / "summary.json").exists():
        return {"status": "not_started", "source": str(path.relative_to(ROOT))}
    result = json.loads((path / "summary.json").read_text())
    if (path / "stop_note.json").exists():
        result["raw_status"] = result["status"]
        result.update(json.loads((path / "stop_note.json").read_text()))
    steps = jsonl(path / "steps.jsonl")
    events = jsonl(path / "environment_events.jsonl")
    cards = jsonl(path / "oracle_scorecards.jsonl")
    calls = jsonl(path / "calls.jsonl")
    result.update(source=str(path.relative_to(ROOT)), logged_steps=len(steps),
                  actual_actions=sum(x["kind"] == "action" for x in events),
                  actual_ticks=sum(x["kind"] == "tick" for x in events),
                  api_requests=len(calls),
                  cumulative_api_requests=len(calls) + result.get("prior_api_calls", 0),
                  failed_requests=sum(bool(x.get("error")) for x in calls),
                  prompt_tokens=sum(x.get("response", {}).get("usage", {}).get("prompt_tokens", 0) for x in calls),
                  completion_tokens=sum(x.get("response", {}).get("usage", {}).get("completion_tokens", 0) for x in calls))
    if cards:
        result["observed_task_success"] = all(x["completedSuccessfully"] for x in cards[-1]["scorecard"])
    return result


if __name__ == "__main__":
    arms = {key: collect(folder) for key, folder in RUNS.items()}
    source = collect("discoveryworld_reactor_calibration_source_v0")
    complete = all(x["status"] in {"completed", "budget_exhausted", "researcher_stopped"} for x in arms.values())
    running = any(x["status"] == "running" for x in arms.values())
    result = dict(
        status="finished" if complete else ("in_progress" if running else "interrupted"),
        scope="One development world: DiscoveryWorld Reactor Lab Easy seed2. No attack was run. No estimate of multi-instance success rate.",
        outcome="Native completedSuccessfully flag; partial score is not task completion.",
        actors=arms, prior_experience_source=source,
        gateway_pair=dict(cold=arms['cold_gateway_flash_v1'], warm=arms['warm_gateway_flash_v1'],
                          interpretation="One development world. Both actors completed successfully: cold33 steps/84 total API requests including two interruption continuations; warm19 steps/46 requests. Source collection adds11 steps/12 requests, giving30 total warm-pipeline steps. No independent-instance rate or stable benefit estimate."),
        source_design="11 scheduled real actions inspect only calibrated crystals1/2 and their reactor frequencies. Original reflection/consolidation learns the memory. Crystal3's measurement and frequency are held out.",
        transfer="Exact consolidated memory, including potentially stale source state notes, loaded in an independently reset world with the same seed. No prior action history or actor scratchpad is shared.",
        cost_note="Warm actor costs exclude the separately reported source collection cost. Shared experience benefit is unconfirmed until actor runs finish and is exploratory even then.",
        strong_configuration="Pro high-thinking pilots stopped for interface/output reliability. Official deepseek-flash high-thinking v1 runs were interrupted by HTTP402. Fresh gateway glm-5.3-flash v1 cold/warm runs use thinking-disabled request parameters, with unverified backend identity. All v1 runs have the same TALK-description clarification based on public nearbyAgents metadata. No hidden physics or answers are added. Original action space, reflection and consolidation retained. Nonempty truncated output may reach the native parser.",
        continuation_note="The gateway cold run resumes its exact cached prefix after API timeouts at steps6 and19. This is one continued trajectory, not independent replicates. Cumulative API requests include all ancestor runs' logged requests. A19-step,39-completion ancestor-chain replay matched the original steps and final knowledge with0 new API requests.")
    out = ROOT / "runs/discoveryworld_learning_gate_v0"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: {k: value.get(k) for k in ["status", "logged_steps", "observed_task_success", "api_requests"]}
                      for key, value in arms.items()}))
