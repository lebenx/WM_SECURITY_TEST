#!/usr/bin/env python3
"""Timestamp-correct, noninteractive shadow DtACI analysis of one fixed trace.

Uses only cached JSONL and the verified native DtACI class. No learned model,
environment, policy, network or Torch is imported. Shadow outputs never reach
the live actor. Equal-count missingness is a retrospective diagnostic.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("native_gate_time_correct", "full_issued_time_correct", "matched_missing_time_correct")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def uniform(domain, lifetime, horizon, target):
    key = json.dumps([domain, 17, lifetime, horizon, target], separators=(",", ":")).encode()
    return (int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11) / (1 << 53)


def native_class(path):
    """Keep original class and pinball helper; replace only its RNG namespace."""
    context = {}

    def choice(values, size=None, replace=True, p=None):
        assert size is None and replace is True
        values = np.asarray(values)
        probabilities = np.ones(len(values)) / len(values) if p is None else np.asarray(p)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        cdf = np.cumsum(probabilities, dtype=np.float64)
        assert cdf[-1] > 0
        cdf /= cdf[-1]
        u = uniform("dtaci-choice-v1", context["lifetime"], context["horizon"], context["target"])
        return values[min(int(np.searchsorted(cdf, u, side="right")), len(values) - 1)]

    class NumpyFacade:
        random = SimpleNamespace(choice=choice)

        def __getattr__(self, name):
            return getattr(np, name)

    parsed = ast.parse(path.read_text())
    kept = [node for node in parsed.body
            if (isinstance(node, ast.ClassDef) and node.name == "DtACI")
            or (isinstance(node, ast.FunctionDef) and node.name == "pinball")
            or (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "vec_zero_min" for t in node.targets))]
    assert len(kept) == 3
    namespace = {"np": NumpyFacade()}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DtACI"], context


def main_analysis(run, out):
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "completed_single_native_clean_episode", "Native trace incomplete"
    protocol = json.loads((run / "frozen_protocol.json").read_text())
    assert summary["scientific_settings"]["pred_interval"] == 1
    assert protocol["shadow_conditions"]["matched_missing_seed"] == 17
    manifest = json.loads(Path(summary["resource_manifest"]["path"]).read_text())
    source = Path(manifest["destination"]) / "dt_aci/one_step_dtai.py"
    expected = next(row for row in manifest["files"] if row["path"] == "dt_aci/one_step_dtai.py")
    data = source.read_bytes()
    assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected["git_blob_sha1"]
    shutil.copy2(source, out / "native_one_step_dtai.py")
    shutil.copy2(__file__, out / "launched_analyzer.py")
    issues = read_rows(run / "issues.jsonl")
    steps = read_rows(run / "physical_steps.jsonl")
    assert len(steps) == summary["outcome"]["steps"]
    assert all(row["after"]["physical_index"] == index + 1 for index, row in enumerate(steps))
    assert [row["done"] for row in steps] == [False] * (len(steps) - 1) + [True]
    assert all(row["before"]["reset_index"] == row["after"]["reset_index"] == 1 for row in steps)
    truth, by_life, eligible, excluded = {}, defaultdict(dict), [], []
    for state in [steps[0]["before"]] + [row["after"] for row in steps]:
        t = state["physical_index"]
        assert state["time"] == t * 0.25
        for h in state["humans"]:
            key = (h["lifetime_id"], t)
            assert key not in truth
            truth[key] = h["position"]
    for row in issues:
        t = row["physical_index"]
        assert row["reset_index"] == 1 and row["time"] == t * 0.25 and t < len(steps)
        for h in row["humans"]:
            lifetime = h["lifetime_id"]
            assert t not in by_life[lifetime]
            if h["issue_valid"] and h["gst_out_mask"]:
                assert h["gst_delivered_relative_exact"], (
                    "GST-to-human assignment invalid", lifetime, t)
            by_life[lifetime][t] = h
    assert len(issues) == len(steps), "Each executed action needs exactly one preceding issue"

    # Choose the first event solely from masks/identity, before residual scoring.
    first = None
    for t in range(len(steps)):
        for lifetime in sorted(by_life):
            history = by_life[lifetime]
            if not all(s in history for s in (t - 2, t - 1, t)):
                continue
            current = history[t]
            masks_ok = (not history[t - 2]["visible"] and history[t - 1]["visible"]
                        and current["visible"] and current["previous_sample_visible"]
                        and current["gst_out_mask"] and current["issue_valid"])
            seen_valid, previous_invalid_after_valid = False, False
            for s in sorted(k for k in history if k < t):
                if history[s]["issue_valid"]:
                    seen_valid = True
                elif seen_valid:
                    previous_invalid_after_valid = True
            if masks_ok and previous_invalid_after_valid:
                first = {"physical_index": t, "time_seconds": t * 0.25, "lifetime_id": lifetime,
                         "human_id": current["human_id"], "selection_uses_residuals": False,
                         "visibility_t_minus2_to_t": [history[s]["visible"] for s in (t-2, t-1, t)],
                         "current_gst_out_mask": True, "current_issue_valid": True,
                         "prior_valid_then_invalid": True}
                break
        if first is not None:
            break
    write_json(out / "first_root_frozen_before_scoring.json", first)

    for lifetime, history in sorted(by_life.items()):
        for issue, h in sorted(history.items()):
            if not h["issue_valid"]:
                continue
            for horizon in range(1, 6):
                target = issue + horizon
                if (lifetime, target) not in truth:
                    excluded.append({"lifetime_id": lifetime, "issue": issue, "horizon": horizon,
                                     "target": target, "reason": "terminal_right_censored" if target > len(steps)
                                     else "same_lifetime_truth_unavailable"})
                    continue
                assert target - 1 in history, "Missing same-lifetime admission event"
                residual = float(np.linalg.norm(np.asarray(h["forecast_world_xy"][horizon], dtype=np.float64)
                                                - np.asarray(truth[(lifetime, target)], dtype=np.float64)))
                eligible.append({"lifetime_id": lifetime, "issue": issue, "target": target,
                                 "horizon": horizon, "actual_gst": h["gst_out_mask"],
                                 "residual_meters": residual,
                                 "native_admitted": history[target - 1]["issue_valid"],
                                 "missingness_priority": uniform("missingness-v1", lifetime, horizon, target)})

    DtACI, choice_context = native_class(source)

    def replay(population, cutoff, selected_lifetime=None):
        groups = defaultdict(list)
        for row in eligible:
            if row["target"] <= cutoff and (population == "dummy_inclusive" or row["actual_gst"]):
                groups[(row["lifetime_id"], row["horizon"])].append(row)
        strata, root_vectors, all_h1_equal = [], {arm: {"raw": [], "clipped": []} for arm in ARMS}, True
        memberships = {}
        for lifetime, history in sorted(by_life.items()):
            if min(history) > cutoff:
                continue
            for horizon in range(1, 6):
                rows = sorted(groups[(lifetime, horizon)], key=lambda r: r["target"])
                n_kept = sum(row["native_admitted"] for row in rows)
                random_kept = {row["target"] for row in sorted(rows, key=lambda r: (r["missingness_priority"], r["target"]))[:n_kept]}
                admitted = {
                    ARMS[0]: [r for r in rows if r["native_admitted"]],
                    ARMS[1]: rows,
                    ARMS[2]: [r for r in rows if r["target"] in random_kept]}
                assert len(admitted[ARMS[0]]) == len(admitted[ARMS[2]])
                omitted = [r for r in rows if not r["native_admitted"]]
                mean = lambda seq: float(np.mean([r["residual_meters"] for r in seq])) if seq else None
                item = {"lifetime_id": lifetime, "horizon": horizon, "eligible": len(rows),
                        "native_retained": n_kept, "native_omitted": len(omitted),
                        "matched_retained": len(random_kept),
                        "native_retained_mean": mean(admitted[ARMS[0]]),
                        "native_omitted_mean": mean(omitted),
                        "matched_retained_mean": mean(admitted[ARMS[2]])}
                if n_kept and omitted:
                    item["omitted_minus_retained_mean"] = item["native_omitted_mean"] - item["native_retained_mean"]
                    item["matched_minus_native_retained_mean"] = item["matched_retained_mean"] - item["native_retained_mean"]
                else:
                    item["omitted_minus_retained_mean"] = item["matched_minus_native_retained_mean"] = None
                q_logs, sequences = {}, {}
                for arm in ARMS:
                    choice_context.update(lifetime=lifetime, horizon=horizon, target="initial")
                    calibrator = DtACI(alpha=0.1, initial_pred=horizon / 10)
                    selected = {r["target"]: r for r in admitted[arm]}
                    q_log = []
                    for t in range(min(history), cutoff + 1):
                        if t in selected:
                            choice_context["target"] = t
                            calibrator.update_true_value(selected[t]["residual_meters"])
                        # Same original prediction-request schedule for all arms.
                        if t in history and history[t]["issue_valid"]:
                            q = float(calibrator.make_prediction())
                            assert np.isfinite(q)
                            q_log.append((t, q, float(np.clip(q, 0, 1))))
                    q_logs[arm] = q_log
                    sequences[arm] = [(r["target"], r["residual_meters"]) for r in admitted[arm]]
                    if lifetime == selected_lifetime:
                        assert q_log and q_log[-1][0] == cutoff
                        root_vectors[arm]["raw"].append(q_log[-1][1])
                        root_vectors[arm]["clipped"].append(q_log[-1][2])
                if horizon == 1:
                    assert n_kept == len(rows)
                    bitwise = lambda values: np.asarray(values, dtype=np.float64).tobytes()
                    h1_ok = (len({bitwise(q_logs[arm]) for arm in ARMS}) == 1
                             and len({bitwise(sequences[arm]) for arm in ARMS}) == 1)
                    assert h1_ok, "h1 clock/RNG/admission negative control failed"
                    all_h1_equal &= h1_ok
                    item["h1_q_and_residuals_bitwise_equal"] = h1_ok
                strata.append(item)
                for row in rows:
                    memberships[(lifetime, horizon, row["target"])] = row["target"] in random_kept
        return {"population": population, "cutoff": cutoff, "strata": strata,
                "h1_all_lifetimes_bitwise_equal": all_h1_equal,
                "selected_lifetime_intervals_h1_to_h5": root_vectors if selected_lifetime is not None else None}, memberships

    np_state, py_state = np.random.get_state(), random.getstate()
    primary, dummy, memberships = None, None, {}
    if first is not None:
        primary, memberships = replay("actual_gst", first["physical_index"], first["lifetime_id"])
        dummy, _ = replay("dummy_inclusive", first["physical_index"], first["lifetime_id"])
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert py_state == random.getstate(), "Shadow analysis consumed native/global RNG"

    gate = {"clean_goal_without_collision": bool(summary["outcome"]["success"] and not summary["outcome"]["collision"]),
            "first_root_exists": first is not None, "actual_omission_supported": False,
            "residual_direction": False, "interval_direction": False,
            "harm_unmeasured": True, "scope": "Natural selection/interval qualification only; native policy remains unrepaired"}
    if primary is not None:
        selected = [r for r in primary["strata"] if r["lifetime_id"] == first["lifetime_id"] and r["horizon"] >= 2]
        supported = [r for r in selected if r["native_retained"] and r["native_omitted"]]
        gate["J_count_only_horizons"] = [r["horizon"] for r in supported]
        gate["actual_omission_supported"] = bool(supported)
        gate["mean_J_omitted_minus_retained_meters"] = float(np.mean([r["omitted_minus_retained_mean"] for r in supported])) if supported else None
        gate["mean_J_matched_minus_native_retained_meters"] = float(np.mean([r["matched_minus_native_retained_mean"] for r in supported])) if supported else None
        gate["residual_direction"] = bool(supported and gate["mean_J_omitted_minus_retained_meters"] > 1e-6
                                         and gate["mean_J_matched_minus_native_retained_meters"] > 1e-6)
        vectors = primary["selected_lifetime_intervals_h1_to_h5"]
        native = np.array(vectors[ARMS[0]]["clipped"])[1:]
        gate["mean_h2_to_h5_full_minus_native_q"] = float(np.mean(np.array(vectors[ARMS[1]]["clipped"])[1:] - native))
        gate["mean_h2_to_h5_matched_minus_native_q"] = float(np.mean(np.array(vectors[ARMS[2]]["clipped"])[1:] - native))
        gate["interval_direction"] = (gate["mean_h2_to_h5_full_minus_native_q"] > 1e-6
                                      and gate["mean_h2_to_h5_matched_minus_native_q"] > 1e-6)
    gate["passed"] = all(gate[k] for k in ("clean_goal_without_collision", "first_root_exists", "actual_omission_supported", "residual_direction", "interval_direction"))
    gate["decision"] = "qualified_for_separate_research_review_not_harm" if gate["passed"] else "stop_fixed_case_candidate_no_replacement_root"
    # Whole-episode counts are descriptive only; they cannot replace the first root.
    whole = defaultdict(lambda: {"eligible": 0, "native_retained": 0})
    for row in eligible:
        if row["actual_gst"]:
            group = whole[(row["lifetime_id"], row["horizon"])]
            group["eligible"] += 1
            group["native_retained"] += int(row["native_admitted"])
    report = {"status": "completed_cache_only_shadow_qualification", "native_run": str(run),
              "native_outcome": summary["outcome"], "first_root": first, "qualification": gate,
              "primary": primary, "dummy_inclusive_diagnostic": dummy,
              "whole_episode_actual_gst_counts_not_primary": [{"lifetime_id": k[0], "horizon": k[1], **v} for k, v in sorted(whole.items())],
              "excluded_unmatured_or_lifetime_changed": excluded,
              "rng_adapter": {"seed": 17, "encoding": "JSON compact [domain,17,lifetime,horizon,target] UTF-8",
                              "uniform": "First SHA256 64 bits >>11 divided by2**53",
                              "choice_domain": "dtaci-choice-v1", "mask_domain": "missingness-v1",
                              "constructor_target": "initial", "choice": "Native probabilities, normalized cumulative sum, right inverse CDF",
                              "global_numpy_and_python_rng_unchanged": True},
              "new_neural_forward_calls": 0, "new_environment_steps": 0,
              "limitations": ["One prerecorded native case with original time-indexing bug; shadows do not drive actions.",
                              "Full simulator labels are privileged. Equal-count selection is retrospective.",
                              "No attacker, action change, collision effect, safety recovery or population claim is tested."]}
    write_json(out / "summary.json", report)
    columns = ["lifetime_id", "issue", "target", "horizon", "actual_gst", "residual_meters",
               "native_admitted", "missingness_priority", "matched_admitted_primary_prefix"]
    with (out / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in eligible:
            writer.writerow({**row, "matched_admitted_primary_prefix": memberships.get((row["lifetime_id"], row["horizon"], row["target"])) if row["actual_gst"] else None})
    print(json.dumps({"status": report["status"], "first_root": first, "qualification": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/gensafenav_feedback_selection_gate_20261015")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.out.resolve() if args.out else run / "shadow_analysis"
    out.mkdir(parents=True, exist_ok=False)
    try:
        main_analysis(run, out)
    except Exception as error:
        write_json(out / "summary.json", {"status": "technical_invalid_retained", "error": repr(error),
                                          "traceback": traceback.format_exc(), "qualification": None})
        raise


if __name__ == "__main__":
    main()
