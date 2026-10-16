"""Native continuations of a fixed historical measurement-failure checkpoint.

The four selectors were frozen by an offline screen using the SAME recorded
public measurements and pre-calibration task feedback. This script recreates
the historical physical state by replaying its exact action prefix in a fresh
native world, then executes the frozen predictions. Fresh radiation noise is
logged but does not replace the fixed historical evidence supplied to selectors.
No hypothetical edited reading is presented as a real measurement.

This is a causal continuation comparison on ONE selected development failure,
not four independent worlds, an autonomous learner, or held-out generalization.
All replay and continuation actions are counted separately and in total.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from pilot_reactor_lazy_calibration import Collector
from pilot_reactor_repair_consensus import VENDOR, save
from pilot_reactor_simple_program import tune


def main(args):
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    case_path = Path(args.screen)/"case_3.json"
    case = json.loads(case_path.read_text())
    source = Path(case["source"])
    historical_result = json.loads((source/"result.json").read_text())
    source_rows = json.loads((source/"protocol_observations.json").read_text())
    targets = [r for r in source_rows if r["observed_calibration_hz"] is None]
    prefix_length = (historical_result["collection_actions"] + historical_result["calibration_actions"]
                     + sum(e["actions"] for e in historical_result["events"] if e["phase"] == "raw"))
    records = [json.loads(x) for x in (source/"public_actions.jsonl").read_text().splitlines()]
    prefix = records[:prefix_length]
    assert prefix[-1]["index"] == prefix_length
    models = case["evaluations"][0]
    assert models["edit_penalty"] == 4
    methods = ["ordinary_refit", "finite_model_elimination", "errors_in_variables", "feedback_evidence_revision"]
    # All method decisions are fixed before any new simulator outcome.
    selections = {method: models[method] for method in methods}
    assert all(selections.values())
    save(output/"plan.json", {"scope": __doc__, "source": str(source),
         "offline_selection_source": str(case_path), "difficulty": historical_result["difficulty"],
         "seed": historical_result["seed"], "prefix_actions": prefix_length,
         "fixed_historical_evidence": True, "fresh_measurements_used_for_selection": False,
         "frozen_selections": selections, "native_actions_budget_per_arm": Collector.action_budget,
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "source_actions_sha256": hashlib.sha256((source/"public_actions.jsonl").read_bytes()).hexdigest(),
         "api_requests": 0, "gpu_work": False})
    (output/"script.py").write_bytes(Path(__file__).read_bytes())
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(VENDOR))
    results = []
    try:
        with (output/"native_stdout.log").open("w", buffering=1) as log, redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for index, method in enumerate(methods):
                folder = output/method
                folder.mkdir()
                api = DiscoveryWorldAPI(threadID=10300+index)
                api.FRAME_DIR = str(folder/"frames")
                assert api.loadScenario("Reactor Lab", historical_result["difficulty"],
                                        randomSeed=historical_result["seed"], numUserAgents=1)
                collector = Collector(api, folder)
                for record in prefix:
                    collector.act(**record["action"])
                # Require the public physical checkpoint to agree before testing.
                # Action-history messages may differ because they contain fresh
                # sensor noise. Never replace these messages with historical text.
                comparison = {}
                for key in ["agentLocation", "inventoryObjects", "accessibleEnvironmentObjects", "dialog_box", "taskProgress"]:
                    comparison[key] = collector.observation[key] == prefix[-1]["observation"][key]
                save(folder/"checkpoint_comparison.json", comparison)
                assert all(comparison.values()), comparison
                predictions = selections[method]["target_predictions_hz"]
                tuning = [tune(collector, row, prediction) for row, prediction in zip(targets, predictions)]
                progress = collector.observation["taskProgress"]
                result = {"method": method, "fixed_selection": selections[method],
                          "checkpoint_public_fields_match": comparison,
                          "replayed_actions": prefix_length,
                          "continuation_actions": collector.actions-prefix_length,
                          "total_actions": collector.actions, "tuning": tuning,
                          "public_task_progress": progress,
                          "completed_successfully": bool(progress) and all(t["completedSuccessfully"] for t in progress)}
                save(folder/"result.json", result)
                results.append(result)
                print(json.dumps({k: result[k] for k in ["method", "completed_successfully", "continuation_actions", "total_actions"]}), file=sys.__stdout__, flush=True)
                del api, collector
        summary = {"status": "completed", "scope": __doc__, "historical_checkpoints": 1,
                   "native_continuations": len(results), "unique_worlds": 1,
                   "results": results, "elapsed_s": time.monotonic()-started,
                   "native_actions": sum(r["total_actions"] for r in results),
                   "replay_actions": sum(r["replayed_actions"] for r in results),
                   "continuation_actions": sum(r["continuation_actions"] for r in results),
                   "api_requests": 0, "gpu_work": False}
        save(output/"summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ["scope", "results"]}), flush=True)
    except Exception as error:
        save(output/"status.json", {"status": "failed", "error": repr(error), "completed_continuations": len(results)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", default="runs/reactor_evidence_revision_v0")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
