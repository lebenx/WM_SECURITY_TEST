"""Test a specific source-learned scalar contrast on native experiment responses.

The researcher supplies the positive-control power-law family and paired design.
This is an operation screen, not autonomous discovery or an official task score.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/DiscoverPhysics/ScienceAgent"))


def acceleration(record):
    response = record["response"]
    t = response["measurement_times"]
    v = response["velocity2"]
    return (v[1][0] - v[0][0]) / (t[1] - t[0])


def exponent(pair):
    # Generic log-linear fit y = c * p2**q with an unknown signed c.
    a, b = [acceleration(r) for r in pair]
    x, y = [r["request"]["p2"] for r in pair]
    assert a * b > 0
    return math.log(abs(b / a)) / math.log(y / x)


def residual(pair, q):
    a, b = [acceleration(r) for r in pair]
    x, y = [r["request"]["p2"] for r in pair]
    w = (y / x) ** (-q)
    return abs(a - w * b) / (abs(a) + abs(w * b))


def main():
    from scienceagent.worlds import get_world

    output = ROOT / "runs/discoverphysics_contrast_transfer_v0"
    output.mkdir(exist_ok=False)
    save = lambda name, value: (output / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    started = time.monotonic()
    initial = json.loads((ROOT / "runs/discoverphysics_interface_prepare_v1/"
                                "requests_responses.json").read_text())
    calibration = {w: [r for r in initial if r["world"] == w]
                   for w in ["gravity", "coulomb_easy"]}
    source_q = exponent(calibration["gravity"])
    # With unrestricted target recalibration, the proposed update has exactly
    # the same estimator and inputs as target-only power-law fitting.
    target_q = {w: exponent(pair) for w, pair in calibration.items()}
    plan = dict(
        source="gravity", targets=list(calibration),
        source_artifact=dict(exponent=source_q, expression="a(m1) - (m2/m1)**(-q) * a(m2)"),
        target_calibration=target_q,
        reused_calibration_experiments=len(initial),
        planned_new_native_experiments=32, noise_std=0.0,
        primary_readout="absolute contrast / sum of absolute weighted responses",
        supplied_priors=["same p2 control correspondence", "scalar positive p2 power-law family",
                        "researcher-selected paired interventions", "initial acceleration estimator"],
        scope="Exploratory operation screen after inspecting environment source; not blind evaluation.",
        no_target_law_access="Only executor.run responses enter fitted exponents; hidden law strings excluded.",
    )
    save("plan.json", plan)
    records, results = [], []
    for world in calibration:
        executor = get_world(world, engine="nbody", noise_std=0.0)["executor"]
        for p1 in [0.7, 1.8]:
            for radius in [4.0, 7.0]:
                for masses in [(0.5, 1.5), (1.5, 3.0)]:
                    pair = []
                    for p2 in masses:
                        request = dict(p1=p1, p2=p2, pos2=[radius, 0.0],
                                       velocity2=[0.0, 0.0],
                                       measurement_times=[0.0, 0.025, 0.05, 0.1, 0.2],
                                       duration=5.0)
                        response = executor.run([request])[0]
                        rec = dict(world=world, request=request, response=response)
                        assert math.isfinite(acceleration(rec))
                        pair.append(rec)
                        records.append(rec)
                        save("requests_responses.json", records)
                    frozen = residual(pair, source_q)
                    recalibrated = residual(pair, target_q[world])
                    target_only = residual(pair, exponent(calibration[world]))
                    results.append(dict(world=world, p1=p1, radius=radius,
                                        p2_pair=list(masses), frozen_source=frozen,
                                        target_recalibrated=recalibrated,
                                        target_only=target_only))
        print(json.dumps(dict(world=world, completed_native_experiments=len(records))), flush=True)
    save("paired_results.json", results)
    by_world = {}
    for world in calibration:
        subset = [r for r in results if r["world"] == world]
        by_world[world] = {
            name: {"mean": sum(r[name] for r in subset) / len(subset),
                   "max": max(r[name] for r in subset)}
            for name in ["frozen_source", "target_recalibrated", "target_only"]}
    summary = dict(
        status="completed", source_exponent=source_q, target_exponents=target_q,
        native_experiments=len(records), reused_calibration_experiments=len(initial),
        heldout_pairs=len(results), by_world=by_world,
        max_recalibrated_vs_target_only_difference=max(
            abs(r["target_recalibrated"] - r["target_only"]) for r in results),
        api_requests=0, gpu_work=False, elapsed_seconds=time.monotonic() - started,
        decision="Do not accept frozen scalar nulling or unrestricted scalar recalibration as "
                 "the new method. Source nulling need not transport; recalibration here is "
                 "identical to ordinary target-only fitting in the supplied family.",
        limitation="Two deterministic native fixtures and researcher-supplied paired power-law "
                   "family only; no autonomous law discovery, new action-interface compiler, "
                   "strong general discovery-agent comparison, or official task score.",
    )
    save("summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
