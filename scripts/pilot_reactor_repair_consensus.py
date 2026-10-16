"""Test a necessary premise of repair-set transfer on native Reactor Lab data.

This is a CPU-only identifiability diagnostic, not a learned repair method or
an end-to-end agent benchmark. Grant the correct affine function family but
not its sensor binding or coefficients. Can the target calibration evidence
make an unseen crystal's frequency identifiable? Compare against the ordinary
version space with exactly the same family and evidence.

Only public UI text and successful native actions enter the data. No scorecard,
hidden crystal attributes, or simulator frequency labels are read. Using the
native UI's JSON renderer skips image rendering, not action/state transitions.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/DiscoveryWorld"
INSTRUMENTS = ("densitometer", "microscope", "radiation meter", "spectrometer", "thermometer")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parse_measurement(name, message):
    if name == "spectrometer":
        values = {f"spectrum_{i}": float(v) for i, v in
                  re.findall(r"Channel (\d+): ([\d.]+)", message)}
        assert len(values) == 5, message
        return values
    patterns = {
        "densitometer": ("density", r"approximately ([\d.]+) grams"),
        "microscope": ("quantum_gap", r"gap of this crystal appears to be ([\d.]+) nm"),
        "radiation meter": ("radiation", r"reports a level of ([\d.]+) micro"),
        "thermometer": ("temperature", r"temperature of ([\d.]+) degrees"),
    }
    key, pattern = patterns[name]
    match = re.search(pattern, message)
    assert match, message
    return {key: float(match[1])}


class PublicCollector:
    expected_crystals = 4
    expected_calibrations = 2

    def __init__(self, api, output):
        self.api, self.output = api, output
        self.known = {}
        self.actions = 0
        self.observation = self.observe()

    def observe(self):
        obs = self.api.ui[0].renderJSON()
        groups = [obs["inventoryObjects"], obs["accessibleEnvironmentObjects"]]
        groups += list(obs["nearbyObjects"]["objects"].values())
        for group in groups:
            for obj in group:
                self.known[obj["uuid"]] = obj
        return obs

    def act(self, **action):
        assert self.actions < 100, "Fixed collection action budget exceeded"
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

    def goto(self, uuid):
        assert uuid in self.known, "Navigation must use an observed object ID"
        return self.act(action="TELEPORT_TO_OBJECT", arg1=uuid)

    def collect(self):
        # The API explicitly exposes these navigation landmarks to agents.
        locations = self.api.listTeleportLocationsDict()
        save(self.output / "public_locations.json", locations)
        for name in sorted(locations):
            self.act(action="TELEPORT_TO_LOCATION", arg1=name)
        instruments = {o["name"]: uid for uid, o in self.known.items()
                       if o["name"] in INSTRUMENTS}
        crystals = sorted((o for o in self.known.values()
                           if o["name"].startswith("quantum crystal ")),
                          key=lambda obj: obj["name"])
        assert set(instruments) == set(INSTRUMENTS), instruments
        assert crystals, "No crystals found through public observations"
        if self.expected_crystals is not None:
            assert len(crystals) == self.expected_crystals, crystals
        # Identify the pre-calibrated pairs from public names and containment.
        calibrations = {}
        for crystal in crystals:
            match = re.search(r"in crystal reactor \(activated\) \[uuid: (\d+)\]",
                              crystal["description"])
            if not match:
                continue
            reactor = int(match[1])
            self.goto(reactor)
            obs = self.act(action="TALK", arg1=reactor)
            dialog = obs["dialog_box"]
            freq = re.search(r"frequence is: ([\d.]+) Hertz", dialog["dialogIn"])
            assert freq, dialog
            calibrations[crystal["uuid"]] = float(freq[1])
            exit_option = next(int(i) for i, value in dialog["dialogOptions"].items()
                               if value == "Exit")
            self.act(chosen_dialog_option_int=exit_option)
        assert calibrations, "No activated reference reactor found"
        if self.expected_calibrations is not None:
            assert len(calibrations) == self.expected_calibrations, calibrations
        # Carry instruments; leave each crystal and reactor in its native place.
        for name in INSTRUMENTS:
            self.goto(instruments[name])
            self.act(action="PICKUP", arg1=instruments[name])
        rows = []
        for crystal in crystals:
            uid = crystal["uuid"]
            self.goto(uid)
            readings, messages = {}, {}
            for name in INSTRUMENTS:
                obs = self.act(action="USE", arg1=instruments[name], arg2=uid)
                message = obs["lastActionMessage"]
                readings.update(parse_measurement(name, message))
                messages[name] = message
            rows.append({"name": crystal["name"], "uuid": uid,
                         "observed_calibration_hz": calibrations.get(uid),
                         "readings": readings, "measurement_messages": messages})
        save(self.output / "observations.json", rows)
        return rows


def analyze(rows):
    """Finite subfamily suffices to refute unanimity in the larger family.

    For each sensor, solve y = a*x+b using the two public references. One
    exact-fit representative is enough to witness disagreement. Roundoff
    uncertainty would enlarge the consistent set, never restore unanimity.
    Sensor readings with nuisance effects may be misspecified; no claim of
    truth coverage is made. Reference-only agreement is not validation.
    """
    calibration = [row for row in rows if row["observed_calibration_hz"] is not None]
    targets = [row for row in rows if row["observed_calibration_hz"] is None]
    assert len(calibration) == len(targets) == 2
    y = np.array([row["observed_calibration_hz"] for row in calibration])
    candidates = []
    for sensor in sorted(calibration[0]["readings"]):
        x = np.array([row["readings"][sensor] for row in calibration])
        if abs(x[1] - x[0]) < 1e-10:
            continue
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        offset = float(y[0] - slope * x[0])
        pred = [slope * row["readings"][sensor] + offset for row in targets]
        # Independent least-squares implementation of the same version space.
        coef = np.linalg.solve(np.column_stack([x, np.ones(2)]), y)
        baseline = [float(np.array([row["readings"][sensor], 1.0]) @ coef)
                    for row in targets]
        assert np.allclose(pred, baseline, atol=1e-7, rtol=1e-10)
        candidates.append({"sensor": sensor, "slope": slope, "offset": offset,
                           "reference_max_error_hz": float(np.max(np.abs(slope*x+offset-y))),
                           "target_predictions_hz": pred,
                           "ordinary_version_space_predictions_hz": baseline,
                           # The controls publicly advertise this physical range.
                           "all_targets_in_public_range": all(0 <= v <= 10000 for v in pred)})
    def summarize_set(group):
        by_target = []
        for index, row in enumerate(targets):
            values = [candidate["target_predictions_hz"][index] for candidate in group]
            spread = max(values) - min(values) if values else None
            # A spread >= 4 Hz rules out any setpoint being within the native
            # strict 2-Hz activation tolerance of every candidate frequency.
            by_target.append({"target": row["name"], "candidate_count": len(values),
                              "minimum_hz": min(values) if values else None,
                              "maximum_hz": max(values) if values else None,
                              "spread_hz": spread,
                              "common_activation_setpoint_exists": bool(values) and spread < 4.0})
        return by_target
    return {"candidates": candidates, "all_candidates": summarize_set(candidates),
            "public_range_filtered": summarize_set(
                [c for c in candidates if c["all_targets_in_public_range"]]),
            "ordinary_version_space_equivalence": True}


def main(args):
    started = time.monotonic()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"status": "running", "seeds": [0, 1, 2], "difficulty": "Normal",
            "candidate_family": "One unknown affine function of one of nine observed sensor channels",
            "grant": "Correct affine structural family supplied as an optimistic premise; not learned from a source",
            "comparison": "Ordinary version space with identical candidate family and target evidence",
            "scope": "Necessary identifiability screen only; no repair learning, source-to-target run, or full task evaluation",
            "decision": "No scale-up if none of the six unseen crystals has a common candidate-supported setpoint",
            "api_requests": 0, "gpu_work": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()}
    save(out / "plan.json", plan)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(VENDOR))
    results = []
    try:
        with (out / "native_stdout.log").open("w", buffering=1) as native_log, redirect_stdout(native_log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for seed in plan["seeds"]:
                folder = out / f"seed_{seed}"
                folder.mkdir()
                api = DiscoveryWorldAPI(threadID=7000 + seed)
                api.FRAME_DIR = str(folder / "frames")
                assert api.loadScenario("Reactor Lab", "Normal", randomSeed=seed, numUserAgents=1)
                collector = PublicCollector(api, folder)
                rows = collector.collect()
                result = {"seed": seed, "native_actions": collector.actions, **analyze(rows)}
                save(folder / "analysis.json", result)
                results.append(result)
                print(json.dumps({"seed_completed": seed, "actions": collector.actions}),
                      file=sys.__stdout__, flush=True)
                del collector, api
        summary = {"status": "completed", "elapsed_s": time.monotonic() - started,
                   "instances": len(results), "unseen_crystals": 2 * len(results),
                   "consensus_setpoints": sum(row["common_activation_setpoint_exists"]
                                              for result in results for row in result["all_candidates"]),
                   "consensus_setpoints_after_public_range_filter": sum(
                       row["common_activation_setpoint_exists"] for result in results
                       for row in result["public_range_filtered"]),
                   "native_actions": sum(result["native_actions"] for result in results),
                   "api_requests": 0, "gpu_work": False, "results": results,
                   "interpretation": "Agreement operator equals ordinary version-space inference; results measure only early target ambiguity, not general impossibility of repair transfer."}
        save(out / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)
    except Exception as error:
        save(out / "status.json", {"status": "failed", "error": repr(error),
                                   "completed_seeds": [r["seed"] for r in results],
                                   "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
