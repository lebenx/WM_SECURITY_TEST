"""Native task-feedback baseline: calibrate only after a failed prediction.

This is an ordinary lazy repair baseline, not a proposed novel algorithm.
It has the same source-informed isolation/minimum operation as the fixed
calibration comparator. Task feedback selects WHEN to use that supplied
operation; it does not discover an instrument, measurement program, or law.
Only public observations and actual actions enter decisions. Native instrument
RNGs are unseeded upstream, so matching world seeds does not pair sensor noise.
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

import numpy as np

from pilot_reactor_acquisition_search import (
    SearchCollector, fixed_calibration, predict_targets, unchanged_candidate,
)
from pilot_reactor_repair_consensus import VENDOR, save
from pilot_reactor_simple_program import tune


class Collector(SearchCollector):
    action_budget = 700


def valid(predictions):
    return all(np.isfinite(p) and 0 <= p <= 10000 for p in predictions)


def execute(collector, rows, arm):
    references = [r for r in rows if r["observed_calibration_hz"] is not None]
    targets = [r for r in rows if r["observed_calibration_hz"] is None]
    events, solved = [], set()
    trigger = None
    if arm == "fixed_calibration":
        trigger = "unconditional"
    else:
        selected = unchanged_candidate(references)
        predictions = predict_targets(collector, targets, selected)
        save(collector.output / "initial_selection.json", selected)
        if not valid(predictions):
            trigger = "prediction_outside_public_actuator_range"
        else:
            for row, prediction in zip(targets, predictions):
                start = collector.actions
                result = tune(collector, row, prediction)
                succeeded = result["public_reactor_name"] == "crystal reactor (activated)"
                events.append({"phase": "raw", "actions": collector.actions-start,
                               "result": result, "activated": succeeded})
                if succeeded:
                    solved.add(row["uuid"])
                elif arm == "lazy_calibration":
                    trigger = "public_reactor_not_activated_after_predicted_setpoint"
                    break
    calibration_actions = 0
    repaired_selection = None
    if arm != "raw" and trigger is not None:
        start = collector.actions
        repaired_selection, predictions = fixed_calibration(collector, rows)
        calibration_actions = collector.actions-start
        save(collector.output / "repair_selection.json", repaired_selection)
        if valid(predictions):
            for row, prediction in zip(targets, predictions):
                if row["uuid"] in solved:
                    continue
                start = collector.actions
                # tune() picks the crystal up first. A previous failed attempt's
                # reactor is then publicly empty and can be reused normally.
                result = tune(collector, row, prediction)
                events.append({"phase": "calibrated", "actions": collector.actions-start,
                               "result": result,
                               "activated": result["public_reactor_name"] == "crystal reactor (activated)"})
    progress = collector.observation["taskProgress"]
    return {"calibration_trigger": trigger, "calibration_actions": calibration_actions,
            "raw_targets_completed_before_calibration": len(solved),
            "calibrated_model": repaired_selection, "events": events,
            "total_actions": collector.actions, "public_task_progress": progress,
            "completed_successfully": bool(progress) and all(t["completedSuccessfully"] for t in progress)}


def main(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [(difficulty, seed) for difficulty in args.difficulties for seed in args.seeds]
    save(output / "plan.json", {
        "cases": cases, "arms": args.arms, "scope": __doc__,
        "action_budget": Collector.action_budget,
        "calibration": "Supplied source-informed isolation plus minimum of 32 readings per crystal; unchanged from the fixed comparator",
        "selection": "Unchanged reference-only finite grammar and coefficient-description ranking",
        "trigger": "Actual public reactor activation failure, or predicted setpoint outside public actuator range",
        "stopping": "At most one full calibration invocation; no candidate search or extra retries after calibrated predictions",
        "counting": "All collection, failed tuning, calibration, navigation, and final tuning actions counted",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "api_requests": 0, "gpu_work": False,
    })
    (output / "script.py").write_bytes(Path(__file__).read_bytes())
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(VENDOR))
    results = []
    try:
        with (output / "native_stdout.log").open("w", buffering=1) as log, redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for difficulty, seed in cases:
                for arm in args.arms:
                    folder = output / f"{difficulty.lower()}_seed_{seed}_{arm}"
                    folder.mkdir()
                    api = DiscoveryWorldAPI(threadID=10100+len(results))
                    api.FRAME_DIR = str(folder / "frames")
                    assert api.loadScenario("Reactor Lab", difficulty, randomSeed=seed, numUserAgents=1)
                    collector = Collector(api, folder)
                    rows = collector.collect()
                    collection_actions = collector.actions
                    result = {"difficulty": difficulty, "seed": seed, "arm": arm,
                              "collection_actions": collection_actions,
                              **execute(collector, rows, arm)}
                    results.append(result)
                    save(folder / "result.json", result)
                    print(json.dumps({k: result[k] for k in ["difficulty", "seed", "arm", "completed_successfully", "total_actions", "calibration_trigger"]}), file=sys.__stdout__, flush=True)
                    del api, collector
        summary = {"status": "completed", "episodes": len(results), "unique_worlds": len(cases),
                   "elapsed_s": time.monotonic()-started, "api_requests": 0, "gpu_work": False,
                   "scope": __doc__, "results": results,
                   "by_arm": {arm: {"successes": sum(r["completed_successfully"] for r in results if r["arm"] == arm),
                                    "episodes": sum(r["arm"] == arm for r in results),
                                    "actions": sum(r["total_actions"] for r in results if r["arm"] == arm),
                                    "calibration_invocations": sum(r["calibration_actions"] > 0 for r in results if r["arm"] == arm)} for arm in args.arms}}
        save(output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ["results", "scope"]}), flush=True)
    except Exception as error:
        save(output / "status.json", {"status": "failed", "error": repr(error), "completed_episodes": len(results)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--difficulties", nargs="+", default=["Normal", "Challenge"])
    parser.add_argument("--arms", nargs="+", choices=["raw", "fixed_calibration", "lazy_calibration"], default=["raw", "fixed_calibration", "lazy_calibration"])
    main(parser.parse_args())
