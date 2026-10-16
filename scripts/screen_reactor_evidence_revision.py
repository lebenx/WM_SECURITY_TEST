"""Compare feedback-directed evidence repair with matched errors-in-variables.

This is an offline algorithm screen on public native records, not a new native
task evaluation. Both repair selectors receive the same finite candidate pool.
Only the feedback selector constrains its choices by recorded failed actuator
settings. A difference is necessary, not sufficient, for a contribution beyond
ordinary noisy-input fitting. We do not assert novelty for either operation.

Candidates perturb at most one isolated reference radiation reading on the
meter's displayed 0.01 grid within that reference's observed sample range, then
refit the existing expression grammar. This supplied local family is neither a
statistical confidence region nor unrestricted observation-model learning.
Unobserved candidate readings are hypotheses, never recorded as measurements.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from pilot_reactor_program_grammar import design, induce, ordering
from pilot_reactor_repair_consensus import save


def predict(candidate, targets):
    model = candidate["model"]
    matrix = design([row["readings"][model["sensor"]] for row in targets], model["terms"])
    return (matrix @ np.asarray(model["coefficients"])).tolist()


def rank(candidate, edit_penalty):
    return (candidate["model"]["description_characters"] + edit_penalty*int(candidate["edit"] is not None),
            *ordering(candidate["model"])[1:], json.dumps(candidate["edit"], sort_keys=True))


def generate(references, measurements):
    candidates = [{"edit": None, "model": model} for model in induce(references)["ranked_programs"]]
    variants = 0
    for index, row in enumerate(references):
        readings = measurements[row["name"]]["values"]
        width = round((max(readings)-min(readings))*100)
        for shift in range(-width, width+1):
            if shift == 0:
                continue
            edited = deepcopy(references)
            before = row["readings"]["radiation"]
            after = round(before+shift*0.01, 2)
            if after < 0:
                continue
            edited[index]["readings"]["radiation"] = after
            variants += 1
            change = {"reference": row["name"], "sensor": "radiation",
                      "observed_minimum": before, "hypothetical_value": after,
                      "display_steps": shift}
            for model in induce(edited)["ranked_programs"]:
                # An edit to an unused feature cannot explain a prediction.
                if model["sensor"] == "radiation":
                    candidates.append({"edit": change, "model": model})
    return candidates, variants


def same_decision(left, right):
    if left is None or right is None:
        return left is right
    return left["edit"] == right["edit"] and left["model"] == right["model"]


def analyze(folder, penalties):
    rows = json.loads((folder/"protocol_observations.json").read_text())
    measurements = {x["name"]: x for x in json.loads((folder/"measurement_diagnostics.json").read_text())}
    result = json.loads((folder/"result.json").read_text())
    references = [r for r in rows if r["observed_calibration_hz"] is not None]
    targets = [r for r in rows if r["observed_calibration_hz"] is None]
    # Only feedback preceding calibration is provided to the selector.
    failures = [e["result"] for e in result["events"] if e["phase"] == "raw" and not e["activated"]]
    assert failures, "Screen needs a real pre-calibration task counterexample"
    candidates, variants = generate(references, measurements)
    target_index = {r["name"]: i for i, r in enumerate(targets)}
    for candidate in candidates:
        values = predict(candidate, targets)
        candidate["target_predictions_hz"] = values
        candidate["valid_predictions"] = all(math.isfinite(p) and 0 <= p <= 10000 for p in values)
        # Source-informed native activation tolerance, supplied identically to
        # the two feedback controls. Actual target frequencies remain hidden.
        candidate["consistent_with_failures"] = candidate["valid_predictions"] and all(
            abs(values[target_index[f["crystal"]]]-f["setpoint_hz"]) >= 2.0 for f in failures)
    valid = [c for c in candidates if c["valid_predictions"]]
    evaluations = []
    for penalty in penalties:
        def select(pool):
            return min(pool, key=lambda c: rank(c, penalty)) if pool else None
        raw = select([c for c in valid if c["edit"] is None])
        elimination = select([c for c in valid if c["edit"] is None and c["consistent_with_failures"]])
        noisy_input = select(valid)
        feedback_repair = select([c for c in valid if c["consistent_with_failures"]])
        evaluations.append({"edit_penalty": penalty, "ordinary_refit": raw,
                            "finite_model_elimination": elimination,
                            "errors_in_variables": noisy_input,
                            "feedback_evidence_revision": feedback_repair,
                            "feedback_changes_repair_decision": not same_decision(noisy_input, feedback_repair)})
    return {"source": str(folder), "references": len(references),
            "target_labels_used": False, "post_calibration_task_feedback_used": False,
            "pre_calibration_failures": failures,
            "input_variants": variants, "candidates": len(candidates),
            "evaluations": evaluations,
            "candidate_pool": candidates}


def main(args):
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(root)/case for root in args.runs for case in
               ["normal_seed_23_lazy_calibration", "challenge_seed_24_lazy_calibration"]]
    save(output/"plan.json", {"sources": [str(p) for p in sources], "scope": __doc__,
         "primary_edit_penalty": 4, "sensitivity_penalties": [0, 2, 8],
         "failure_tolerance_hz": 2.0,
         "comparison": "Identical candidate generation and ranking; only actual earlier failed-setting constraints differ",
         "screen_question": "Does task feedback change the proposed reobservation or model beyond the matched noisy-input baseline?",
         "api_requests": 0, "gpu_work": False, "new_native_actions": 0,
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/"script.py").write_bytes(Path(__file__).read_bytes())
    results = []
    for index, source in enumerate(sources):
        result = analyze(source, [4, 0, 2, 8])
        save(output/f"case_{index}.json", result)
        result.pop("candidate_pool")
        results.append(result)
        print(json.dumps({"source": str(source), "candidates": result["candidates"],
                          "primary_feedback_changes_repair_decision": result["evaluations"][0]["feedback_changes_repair_decision"]}), flush=True)
    summary = {"status": "completed", "scope": __doc__, "datasets": len(results),
               "elapsed_s": time.monotonic()-started, "api_requests": 0,
               "gpu_work": False, "new_native_actions": 0, "results": results,
               "primary_feedback_changed_decisions": sum(r["evaluations"][0]["feedback_changes_repair_decision"] for r in results),
               "all_penalties_feedback_changed_decisions": sum(e["feedback_changes_repair_decision"] for r in results for e in r["evaluations"])}
    save(output/"summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["scope", "results"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", nargs="+", default=["runs/reactor_lazy_calibration_v0", "runs/reactor_lazy_calibration_repeat_v0"])
    main(parser.parse_args())
