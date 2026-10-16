"""Execute a simple coefficient-description baseline with the same collection.

All arms are granted an affine structural family. This baseline selects the
shortest decimal coefficient representation consistent with public reference
measurements (0.005 Hz rounding tolerance), then actually tunes the reactors.
Measurements are freshly collected because native instruments have unseeded
background noise. No hidden formulas or target frequency labels are supplied.
This is a deterministic diagnostic baseline, not a proposed novel learner.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from pilot_reactor_repair_consensus import ROOT, VENDOR, PublicCollector, save


def shortest_program(rows):
    references = [r for r in rows if r["observed_calibration_hz"] is not None]
    targets = [r for r in rows if r["observed_calibration_hz"] is None]
    programs = []
    for sensor in sorted(references[0]["readings"]):
        x0, x1 = [r["readings"][sensor] for r in references]
        y0, y1 = [r["observed_calibration_hz"] for r in references]
        if abs(x1-x0) < 1e-10:
            continue
        a = (y1-y0)/(x1-x0)
        b = y0-a*x0
        # Enumerate decimal encodings; this is an explicitly limited search,
        # not a claim of globally minimum description length over all programs.
        astrings = {format(a, f".{n}g") for n in range(1, 13)}
        bstrings = {format(b, f".{n}g") for n in range(1, 13)}
        candidates = []
        for sa in astrings:
            for sb in bstrings:
                aa, bb = float(sa), float(sb)
                error = max(abs(aa*x0+bb-y0), abs(aa*x1+bb-y1))
                predictions = [aa*r["readings"][sensor]+bb for r in targets]
                if error <= 0.005 + 1e-8 and all(0 <= y <= 10000 for y in predictions):
                    candidates.append({"sensor": sensor, "a": aa, "b": bb,
                                       "encoded_a": sa, "encoded_b": sb,
                                       "coefficient_characters": len(sa)+len(sb),
                                       "reference_max_error_hz": error,
                                       "target_predictions_hz": predictions})
        if candidates:
            programs.append(min(candidates, key=program_order))
    assert programs, "No eligible affine program"
    programs.sort(key=program_order)
    return {"selected": programs[0], "ranked_sensor_programs": programs,
            "description_length_gap": programs[1]["coefficient_characters"]
            - programs[0]["coefficient_characters"] if len(programs) > 1 else None}


def program_order(candidate):
    return (candidate["coefficient_characters"], candidate["reference_max_error_hz"],
            candidate["sensor"], candidate["encoded_a"], candidate["encoded_b"])


class Continuation(PublicCollector):
    def act(self, **action):
        assert self.actions < 180, "Fixed baseline action budget exceeded"
        result = self.api.performAgentAction(0, action)
        self.api.tick()
        obs = self.observe()
        self.actions += 1
        with (self.output / "public_actions.jsonl").open("a") as handle:
            handle.write(json.dumps({"index": self.actions, "action": action,
                                     "result": result, "observation": obs}) + "\n")
        assert result["success"], (action, result, obs["lastActionMessage"])
        self.observation = obs
        return obs


def tune(collector, crystal, prediction):
    collector.goto(crystal["uuid"])
    collector.act(action="PICKUP", arg1=crystal["uuid"])
    empty = sorted((uid for uid, obj in collector.known.items()
                    if obj["name"] == "crystal reactor (no crystal present)"))
    assert empty, "No publicly observed empty reactor"
    reactor = empty[0]
    collector.goto(reactor)
    collector.act(action="PUT", arg1=crystal["uuid"], arg2=reactor)
    obs = collector.act(action="TALK", arg1=reactor)
    setpoint = int(math.floor(prediction + 0.5))
    for _ in range(40):
        dialog = obs["dialog_box"]
        current = float(re.search(r"frequence is: ([\d.]+) Hertz", dialog["dialogIn"])[1])
        difference = setpoint - int(current)
        if difference == 0:
            exit_option = next(int(i) for i, text in dialog["dialogOptions"].items() if text == "Exit")
            collector.act(chosen_dialog_option_int=exit_option)
            break
        step = next(step for step in [1000, 100, 10, 1] if step <= abs(difference))
        option_text = f"{'Increase' if difference > 0 else 'Decrease'} frequency by {step} Hz"
        option = next(int(i) for i, text in dialog["dialogOptions"].items() if text == option_text)
        obs = collector.act(chosen_dialog_option_int=option)
    else:
        raise RuntimeError("Native tuning budget exceeded")
    return {"crystal": crystal["name"], "reactor_uuid": reactor,
            "prediction_hz": prediction, "setpoint_hz": setpoint,
            "public_reactor_name": collector.known[reactor]["name"]}


def main(args):
    started = time.monotonic()
    source, out = Path(args.data).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"status": "running", "collection_protocol_run": str(source.relative_to(ROOT)),
            "seeds": [0, 1, 2], "difficulty": "Normal",
            "family": "One affine function of one measured sensor channel",
            "selection": "Minimum decimal coefficient string length among reference-consistent programs; all sensors have equal binding cost",
            "coefficient_search": "Each solved coefficient formatted with 1 through 12 significant digits",
            "calibration_tolerance_hz": 0.005, "target_rounding": "nearest integer, half up",
            "measurement_policy": "Fresh native readings; exact action sequence matches the prior collection, observations may differ because instrument RNG is unseeded upstream",
            "source_transfer": False, "api_requests": 0, "gpu_work": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out / "plan.json", plan)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(VENDOR))
    results = []
    try:
        with (out / "native_stdout.log").open("w", buffering=1) as log, redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for seed in plan["seeds"]:
                folder = out / f"seed_{seed}"
                folder.mkdir()
                data = source / f"seed_{seed}"
                api = DiscoveryWorldAPI(threadID=8000 + seed)
                api.FRAME_DIR = str(folder / "frames")
                assert api.loadScenario("Reactor Lab", "Normal", randomSeed=seed, numUserAgents=1)
                collector = Continuation(api, folder)
                rows = collector.collect()
                collected = collector.actions
                prior_actions = [json.loads(line)["action"] for line in
                                 (data / "public_actions.jsonl").read_text().splitlines()]
                fresh_actions = [json.loads(line)["action"] for line in
                                 (folder / "public_actions.jsonl").read_text().splitlines()]
                assert prior_actions == fresh_actions, "Collection curriculum changed"
                program = shortest_program(rows)
                save(folder / "program.json", program)
                targets = [row for row in rows if row["observed_calibration_hz"] is None]
                tuning = [tune(collector, row, pred) for row, pred in zip(
                    targets, program["selected"]["target_predictions_hz"])]
                progress = collector.observation["taskProgress"]
                result = {"seed": seed, "selected_program": program["selected"],
                          "collection_actions": collected, "total_actions": collector.actions,
                          "collection_action_sequence_matches": True,
                          "tuning_actions": collector.actions - collected, "tuning": tuning,
                          "public_task_progress": progress,
                          "completed_successfully": bool(progress) and all(
                              task["completedSuccessfully"] for task in progress)}
                save(folder / "result.json", result)
                results.append(result)
                print(json.dumps({"seed": seed, "success": result["completed_successfully"],
                                  "actions": collector.actions}), file=sys.__stdout__, flush=True)
                del collector, api
        summary = {"status": "completed", "elapsed_s": time.monotonic()-started,
                   "instances": len(results),
                   "tasks_completed": sum(r["completed_successfully"] for r in results),
                   "reactors_activated": sum(t["public_reactor_name"] == "crystal reactor (activated)"
                                             for r in results for t in r["tuning"]),
                   "total_native_actions": sum(r["total_actions"] for r in results),
                   "api_requests": 0, "gpu_work": False, "results": results,
                   "scope": "Three development instances, known affine family, scripted data collection/navigation; no LLM-agent or source-transfer result."}
        save(out / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)
    except Exception as error:
        save(out / "status.json", {"status": "failed", "error": repr(error),
                                   "completed_seeds": [r["seed"] for r in results]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
