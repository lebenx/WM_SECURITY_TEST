"""Small target-only program-search baseline on native Reactor Lab.

Choose both sensor and expression from a declared grammar using public reference
measurements. No difficulty, seed, simulator formula, target label, or source
episode is passed to the induction function. This is an ordinary finite MDL
baseline, not a novel algorithm or unrestricted symbolic regression. Navigation,
instrument names, parsers, and grammar are supplied by the researcher.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pilot_reactor_repair_consensus import ROOT, VENDOR, PublicCollector, save
from pilot_reactor_simple_program import tune


GRAMMAR = {
    "constant": (["1"], 1),
    "affine": (["1", "x"], 3),
    "square_affine": (["1", "x^2"], 4),
    "cube_affine": (["1", "x^3"], 4),
    "sqrt_affine": (["1", "sqrt(x)"], 4),
    "log_affine": (["1", "log(x)"], 4),
    "reciprocal_affine": (["1", "1/x"], 4),
    "quadratic": (["1", "x", "x^2"], 6),
    "cubic": (["1", "x", "x^2", "x^3"], 9),
}
TOLERANCE_HZ = 0.005 + 1e-8


def design(values, terms):
    x = np.asarray(values, dtype=float)
    basis = {"1": lambda: np.ones_like(x), "x": lambda: x,
             "x^2": lambda: x*x, "x^3": lambda: x*x*x,
             "sqrt(x)": lambda: np.sqrt(x), "log(x)": lambda: np.log(x),
             "1/x": lambda: 1/x}
    with np.errstate(all="ignore"):
        return np.column_stack([basis[term]() for term in terms])


def encodings(value):
    strings = {format(value, f".{n}g") for n in range(1, 13)}
    strings.update(format(value, f".{n}f") for n in range(7))
    # Remove redundant trailing zeroes only in ordinary decimal notation.
    strings = {s.rstrip("0").rstrip(".") if "." in s and "e" not in s else s
               for s in strings}
    return sorted(strings, key=lambda s: (len(s), s))


def ordering(program):
    return (program["description_characters"], program["reference_max_error_hz"],
            program["sensor"], program["family"], program["encoded_coefficients"])


def induce(rows):
    references = [r for r in rows if r["observed_calibration_hz"] is not None]
    targets = [r for r in rows if r["observed_calibration_hz"] is None]
    y = np.array([r["observed_calibration_hz"] for r in references])
    programs, fits = [], []
    for sensor in sorted(references[0]["readings"]):
        rx = [r["readings"][sensor] for r in references]
        tx = [r["readings"][sensor] for r in targets]
        for family, (terms, structure_cost) in GRAMMAR.items():
            if len(terms) > len(references):
                continue
            matrix, target_matrix = design(rx, terms), design(tx, terms)
            if not (np.isfinite(matrix).all() and np.isfinite(target_matrix).all()):
                continue
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
            if rank < len(terms):
                continue
            fit_error = float(np.max(np.abs(matrix @ coefficients - y)))
            fits.append({"sensor": sensor, "family": family, "max_error_hz": fit_error})
            if fit_error > TOLERANCE_HZ:
                continue
            choices = list(product(*(encodings(c) for c in coefficients)))
            candidates = np.asarray([[float(s) for s in choice] for choice in choices])
            errors = np.max(np.abs(candidates @ matrix.T - y), axis=1)
            predictions = candidates @ target_matrix.T
            valid = np.flatnonzero((errors <= TOLERANCE_HZ) &
                                   np.all((predictions >= 0) & (predictions <= 10000), axis=1))
            if not len(valid):
                continue
            i = min(valid, key=lambda j: (sum(map(len, choices[j])), errors[j], choices[j]))
            programs.append({"sensor": sensor, "family": family, "terms": terms,
                             "coefficients": candidates[i].tolist(),
                             "encoded_coefficients": list(choices[i]),
                             "description_characters": structure_cost + sum(map(len, choices[i])),
                             "reference_max_error_hz": float(errors[i]),
                             "target_predictions_hz": predictions[i].tolist()})
    programs.sort(key=ordering)
    return {"selected": programs[0] if programs else None, "ranked_programs": programs,
            "least_squares_fits": fits, "references": len(references), "targets": len(targets),
            "description_length_gap": programs[1]["description_characters"] -
                programs[0]["description_characters"] if len(programs) > 1 else None}


class Collector(PublicCollector):
    expected_crystals = None
    expected_calibrations = None

    def act(self, **action):
        assert self.actions < 260, "Fixed per-instance action budget exceeded"
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


def main(args):
    started, out = time.monotonic(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"seeds": args.seeds, "difficulties": args.difficulties,
            "grammar": GRAMMAR, "reference_tolerance_hz": TOLERANCE_HZ,
            "fit": "Full-rank least squares; no underdetermined fits; decimal encodings 1..12 significant digits and 0..6 decimal places",
            "selection": "Minimum coefficient text length plus declared expression cost, then reference error; equal sensor binding cost",
            "range": [0, 10000], "action_budget": 260, "source_transfer": False,
            "learner_input": "Only public calibration setpoints and instrument readings; target readings have no frequency labels",
            "scope": __doc__, "api_requests": 0, "gpu_work": False,
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
        with (out / "native_stdout.log").open("w", buffering=1) as log, redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for difficulty in args.difficulties:
                for seed in args.seeds:
                    folder = out / f"{difficulty.lower()}_seed_{seed}"
                    folder.mkdir()
                    api = DiscoveryWorldAPI(threadID=9000 + len(results))
                    api.FRAME_DIR = str(folder / "frames")
                    assert api.loadScenario("Reactor Lab", difficulty, randomSeed=seed, numUserAgents=1)
                    collector = Collector(api, folder)
                    rows = collector.collect()
                    collection_actions = collector.actions
                    inference_start = time.monotonic()
                    model = induce(rows)
                    model["elapsed_s"] = time.monotonic() - inference_start
                    save(folder / "program.json", model)
                    targets = [r for r in rows if r["observed_calibration_hz"] is None]
                    selected = model["selected"]
                    tuning = [tune(collector, row, pred) for row, pred in zip(
                        targets, selected["target_predictions_hz"])] if selected else []
                    progress = collector.observation["taskProgress"]
                    result = {"difficulty": difficulty, "seed": seed,
                              "selected": selected, "inference_elapsed_s": model["elapsed_s"],
                              "references": model["references"], "targets": len(targets),
                              "collection_actions": collection_actions, "total_actions": collector.actions,
                              "tuning": tuning, "public_task_progress": progress,
                              "status": "evaluated" if selected else "no_reference_consistent_program",
                              "completed_successfully": bool(progress) and all(t["completedSuccessfully"] for t in progress)}
                    save(folder / "result.json", result)
                    results.append(result)
                    print(json.dumps({k:result[k] for k in ["difficulty", "seed", "status", "completed_successfully", "total_actions"]}),
                          file=sys.__stdout__, flush=True)
                    del collector, api
        summary = {"status": "completed", "instances": len(results),
                   "tasks_completed": sum(r["completed_successfully"] for r in results),
                   "reactors_activated": sum(t["public_reactor_name"] == "crystal reactor (activated)"
                                              for r in results for t in r["tuning"]),
                   "target_crystals": sum(r["targets"] for r in results),
                   "native_actions": sum(r["total_actions"] for r in results),
                   "elapsed_s": time.monotonic() - started, "api_requests": 0,
                   "gpu_work": False, "results": results,
                   "scope": "Finite grammar and scripted public collection/navigation. No new learning method, unrestricted discovery, source-transfer, or LLM-agent result."}
        save(out / "summary.json", summary)
        print(json.dumps({k:v for k,v in summary.items() if k != "results"}), flush=True)
    except Exception as error:
        save(out / "status.json", {"status": "failed", "error": repr(error),
                                   "completed_instances": len(results), "elapsed_s": time.monotonic()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--difficulties", nargs="+", choices=["Normal", "Challenge"], default=["Normal", "Challenge"])
    main(parser.parse_args())
