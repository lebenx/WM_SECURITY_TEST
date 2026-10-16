#!/usr/bin/env python3
"""Audit finite candidate utilities and recompute descriptive leverage from caches.

No model, environment, checkpoint, network, or third-party library is loaded.
The adapters deliberately admit complete fixed candidate banks, not arbitrary
closed-loop episode contrasts. New adapters require an explicit utility contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/decision_leverage_paper"
COSMOS = "runs/cosmos_first_candidate_utility_recovery_analysis_v1/analysis.json"
DINO = "runs/dino_action_saturation_gate_20261015"
WORLD = "runs/worldagen_prediction_contract_cpu_v2/summary.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_candidate_bounds(utilities, baseline_index=None, higher_is_better=True):
    """Bound outputs which select an identity in this exact finite action bank.

    Passing an elite mean as an index is invalid: a mean of controls generally
    leaves the bank, and its physical utility need not interpolate utilities.
    """
    require(bool(utilities), "Empty candidate bank")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in utilities),
            "Nonfinite or nonnumeric true utility")
    if baseline_index is not None:
        require(type(baseline_index) is int and 0 <= baseline_index < len(utilities),
                "Selector output is not a member of the audited candidate bank")
    low, high = min(utilities), max(utilities)
    chosen = None if baseline_index is None else utilities[baseline_index]
    return {"minimum": low, "maximum": high, "leverage": high - low,
            "baseline_value": chosen,
            "baseline_harm_headroom": None if chosen is None else
            (chosen - low if higher_is_better else high - chosen),
            "uniform_value": sum(utilities) / len(utilities)}


class Inputs:
    def __init__(self, root):
        self.root = root
        self.bindings = {}

    def relative(self, path):
        # Historical records contain the original workspace's absolute paths.
        path = str(path)
        marker = "/WM_Security/"
        if marker in path:
            path = path.split(marker, 1)[1]
        p = Path(path)
        if p.is_absolute():
            p = p.relative_to(self.root)
        require(".." not in p.parts, "Source escaped repository")
        return p.as_posix()

    def bytes(self, path, expected_hash=None):
        name = self.relative(path)
        raw = (self.root / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_hash is not None:
            require(digest == expected_hash, "Historical artifact binding differs: " + name)
        self.bindings[name] = {"path": name, "bytes": len(raw), "sha256": digest}
        return raw

    def json(self, path):
        return json.loads(self.bytes(path))

    def lines(self, path, expected_hash=None):
        return [json.loads(line) for line in self.bytes(path, expected_hash).splitlines() if line.strip()]


def entry(system, root, cluster, candidate, metric, value, higher, source, baseline=None):
    return {"system": system, "root": root, "cluster": cluster, "candidate": candidate,
            "metric": metric, "value": float(value), "higher_is_better": higher,
            "utility_source": source, "baseline_index": baseline}


def cosmos(inputs):
    doc = inputs.json(COSMOS)
    # Reuse the prior arithmetic verifier, then independently reopen real traces.
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_cosmos_first_choice_bounds as old
    old_result = old.analyze(inputs.root / COSMOS)
    require(old_result == inputs.json("reports/cosmos_first_choice_identifiability_bounds_20261015.json"),
            "Existing Cosmos bound report no longer reproduces")
    audit_by_key = {(a["task_id"], a["init_id"], a["forced_candidate_index"]): a
                    for a in doc["episode_audits"]}
    require(len(audit_by_key) == 60, "Incomplete or repeated Cosmos branches")
    rows = []
    for pair in doc["pairs"]:
        task, init = pair["task_id"], pair["init_id"]
        for candidate in (0, 1):
            audit = audit_by_key[task, init, candidate]
            parity = audit["independent_source_comparison"]
            require(parity["first_full_candidate_set_exact"] and parity["initial_and_settled_state_exact"],
                    "Cosmos fixed-state/candidate parity failed")
            if candidate == pair["source_native_selected_index"]:
                require(parity["selected_arm_whole_trajectory_exact"], "Native source trajectory parity failed")
            source = inputs.relative(audit["artifacts"]["trace"]["path"])
            trace = inputs.lines(source, audit["artifacts"]["trace"]["sha256"])
            require([r["step"] for r in trace] == list(range(1, len(trace) + 1)), "Cosmos trace gap")
            success = int(any(r["task_success"] for r in trace))
            require(not any(r["task_success"] for r in trace[:-1]), "Trace continued after success")
            capped = len(trace) if success else 300
            require(success == pair[f"candidate{candidate}_success"] == audit["success"], "Success mismatch")
            require(capped == pair[f"candidate{candidate}_capped_steps"] == audit["capped_steps"], "Capped cost mismatch")
            require(len(trace) == pair[f"candidate{candidate}_executed_steps"] == audit["executed_steps"], "Length mismatch")
            for metric, value, higher in [("success", success, True), ("capped_steps", capped, False)]:
                rows.append(entry("Cosmos", f"task{task:02d}_init{init:02d}", f"task{task:02d}",
                                  candidate, metric, value, higher, source, pair["source_native_selected_index"]))
    info = {"system": "Cosmos", "status": "included_native_first_choice_cache", "roots": 30,
            "candidates_per_root": 2, "physical_branches": 60, "clusters": 10,
            "unit": "task/init root; three initializations within each of ten task clusters",
            "independence": "Fixed development roots, not a random population sample; no independent-replication claim.",
            "candidate_contract": "Same two first-16-control sequences; then original-Goal fixed-seed native continuation.",
            "baseline": "recorded native first-choice index; later native continuation unchanged",
            "utility_contract": "Environment task success; actual controls to success, failure capped at300.",
            "validation": "All60 trace hashes and task utilities reopened; prior60-branch state/candidate parity audit checked; existing bound analyzer rerun.",
            "limitation": "Post-interruption amended development cache. Not a bound for later interventions, different actions, stochastic repetitions, or an elite-mean output.",
            "source": COSMOS}
    return rows, info


def dino(inputs):
    doc = inputs.json(DINO + "/summary.json")
    require(doc["status"] == "completed_cpu_fixed_bank_gate", "DINO cache incomplete")
    require(doc["snapshot"]["initial_pixels_and_proprio_exact"] and doc["snapshot"]["historical_source_prefix_verified"],
            "DINO root parity absent")
    require(all(r["exact_to_original"] for r in doc["replays"] if r["representation"] != "original"),
            "DINO physical alias equivalence absent")
    require(len(doc["utilities"]) == len(doc["bank"]["candidate_names"]) == 5, "DINO bank incomplete")
    baseline = doc["forecasts"]["original"]["selected_candidate"]
    require(baseline == min(range(5), key=lambda i: (doc["forecasts"]["original"]["native_costs"][i], i)),
            "DINO selected-index mismatch")
    rows = []
    for candidate in range(5):
        source = f"{DINO}/candidate{candidate}/original/trace.jsonl"
        trace = inputs.lines(source)
        require(len(trace) == 30 and [x["policy_step"] for x in trace] == list(range(1, 31)), "DINO trace gap")
        for horizon, name in [(15, "native_execution_prefix15"), (30, "primary30")]:
            t, stored = trace[:horizon], doc["utilities"][candidate][name]
            values = {"terminal_native_reward": t[-1]["reward"],
                      "summed_native_reward": sum(x["reward"] for x in t),
                      "final_success": t[-1]["success"], "ever_success": any(x["success"] for x in t),
                      "final_reach_distance": t[-1]["reach_distance"]}
            for key, value in values.items():
                require(value == stored[key], f"DINO utility mismatch: {candidate}/{name}/{key}")
                rows.append(entry("DINO-WM", "reach_seed1", "reach_seed1", candidate,
                                  f"{key}_h{horizon}", value, key != "final_reach_distance", source, baseline))
    info = {"system": "DINO-WM", "status": "included_added_fixed_bank_selector", "roots": 1,
            "candidates_per_root": 5, "physical_branches": 5, "clusters": 1,
            "unit": "one seed1 reach root; all five action sequences replayed from that same snapshot",
            "independence": "Candidate source-planner seeds1/2/3 are not separate outcome roots;15/30 horizons share trajectories.",
            "candidate_contract": "Frozen30-control open-loop bank: three historical native prefixes, zero, and sign reversal.",
            "baseline": "original forecast argmin on the added bank; NOT native CEM output or complete task policy",
            "utility_contract": "Actual environment reward sum, final reward, final/ever success, final reach distance at15 and30 controls.",
            "validation": "All5 original30-step traces reopened; all reported utilities recomputed; snapshot and equivalence audit checked.",
            "limitation": "One development bank. No CI, population ASR, native CEM, or complete task claim. The original30-step oracle competence gate failed and remains failed.",
            "source": DINO + "/summary.json"}
    return rows, info


def worldagen(inputs):
    doc = inputs.json(WORLD)
    audit = inputs.json("reports/worldagen_corrected_decision_evidence_audit_20261015.json")
    require(doc["status"] == "completed_prediction_contract_gate" and doc["forecast_horizon"] == 3,
            "Wrong corrected WorldAgen cache")
    require(doc["world_query_goal_only"] and doc["environment_goal_and_bddl_fixed"]
            and not doc["policy_output_consumed"], "WorldAgen branch contract changed")
    require(len(doc["rows"]) == 3 and audit["independent_source_episodes"] == 1, "WorldAgen root count changed")
    rows = []
    for root in doc["rows"]:
        chunk = root["chunk_index"]
        commit = inputs.json(f"runs/worldagen_prediction_contract_cpu_v2/chunk_{chunk:03d}/prediction_commitment.json")
        require(commit["all_forecasts_saved_before_any_target_replay"], "WorldAgen targets available before commitment")
        require(len(root["candidate_names"]) == 2 and len(root["replays"]) == 4, "WorldAgen incomplete bank")
        for candidate in (0, 1):
            reps = sorted([r for r in root["replays"] if r["candidate"] == candidate], key=lambda r: r["repetition"])
            require([r["repetition"] for r in reps] == [0, 1], "WorldAgen repeated/missing repetitions")
            require(all(r["initial_exact"] and r["future_exact"] for r in reps), "WorldAgen replay parity failed")
            require(reps[0]["future_signature"] == reps[1]["future_signature"], "WorldAgen future state differs")
            for r in reps:
                record = next(x for x in audit["physical_replays"] if
                              (x["chunk_index"], x["candidate"], x["repetition"]) == (chunk, candidate, r["repetition"]))
                require(r["reward_sum"] == record["reward_sum"] and
                        r["success_after_each_action"] == record["success_after_each_action"], "WorldAgen audit differs")
            require(reps[0]["reward_sum"] == reps[1]["reward_sum"] and
                    reps[0]["success_after_each_action"] == reps[1]["success_after_each_action"], "Unequal utility repeats")
            for metric, value in [("success_h3", any(reps[0]["success_after_each_action"])),
                                  ("reward_sum_h3", reps[0]["reward_sum"])]:
                rows.append(entry("WorldAgen", f"task00_init00_chunk{chunk:03d}", "task00_init00",
                                  candidate, metric, value, True, WORLD, 0))
    info = {"system": "WorldAgen", "status": "included_local_utility_diagnostic_only", "roots": 3,
            "candidates_per_root": 2, "physical_branches": 6, "replay_repetitions_per_branch": 2, "clusters": 1,
            "unit": "three points on one recorded successful native episode; not three independent episodes",
            "independence": "One source trajectory; repetitions establish deterministic parity, not new independent samples.",
            "candidate_contract": "Corrected codec: recorded three-action chunk and precommitted x perturbation; original task fixed.",
            "baseline": "candidate0 recorded native action chunk; no forecast-driven policy was executed",
            "utility_contract": "Recorded simulator reward sum and any success during exactly three physical controls.",
            "validation": "Source replay records and independent audit agree; repeat full future signatures and commitment flags checked. Summary-level utility records, not newly rerun simulator.",
            "limitation": "Zero three-step utility span says nothing about longer continuation. Old pre-correction30-step planners are excluded. This row does not establish native forecast-to-action mediation.",
            "source": WORLD}
    return rows, info


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["system"], row["metric"]), {}).setdefault(row["root"], []).append(row)
    root_stats, aggregate, clusters = [], [], []
    for (system, metric), roots in sorted(groups.items()):
        current = []
        for root, bank in sorted(roots.items()):
            bank = sorted(bank, key=lambda r: r["candidate"])
            require([r["candidate"] for r in bank] == list(range(len(bank))), "Incomplete bank IDs")
            require(len({(r["higher_is_better"], r["baseline_index"], r["cluster"]) for r in bank}) == 1,
                    "Root contract inconsistent")
            stats = finite_candidate_bounds([r["value"] for r in bank], bank[0]["baseline_index"], bank[0]["higher_is_better"])
            stats.update(system=system, metric=metric, root=root, cluster=bank[0]["cluster"],
                         candidates=len(bank), higher_is_better=bank[0]["higher_is_better"],
                         baseline_index=bank[0]["baseline_index"], candidate_values=[r["value"] for r in bank])
            current.append(stats)
        root_stats.extend(current)
        for cluster in sorted({r["cluster"] for r in current}):
            part = [r for r in current if r["cluster"] == cluster]
            clusters.append(aggregate_group(system, metric, part, cluster))
        aggregate.append(aggregate_group(system, metric, current))
    return root_stats, aggregate, clusters


def aggregate_group(system, metric, rows, cluster=None):
    result = {"system": system, "metric": metric, "roots": len(rows),
              "clusters": len({r["cluster"] for r in rows}), "higher_is_better": rows[0]["higher_is_better"],
              "candidate_counts": sorted({r["candidates"] for r in rows}),
              "exact_tie_roots": sum(r["leverage"] == 0 for r in rows)}
    if cluster is not None:
        result["cluster"] = cluster
    for key in ("minimum", "maximum", "leverage", "baseline_value", "baseline_harm_headroom", "uniform_value"):
        result[key + "_total"] = sum(r[key] for r in rows)
        result[key + "_mean"] = result[key + "_total"] / len(rows)
    # Identity-wise fixed policies are meaningful only for common indexed banks.
    k = min(r["candidates"] for r in rows)
    result["fixed_candidate_totals"] = [sum(r["candidate_values"][i] for r in rows) for i in range(k)]
    return result


def exclusions(inputs):
    specs = [
        ("TD-MPC2", "reports/tdmpc_nullspace_action_logs_decision_20261015.json",
         "Full episode returns exist for different adapted models, not every candidate from an identical frozen root/bank. Do not use model Q scores as true utility."),
        ("SafeDreamer", "reports/safedreamer_mode_mediator_gate_20261016.json",
         "First-query imagined candidate scores and elite-mean controls only; zero physical steps. Returned elite mean is not a candidate identity."),
        ("GenSafeNav", "reports/gensafenav_feedback_selection_decision_20261015.json",
         "Native Clean trajectory and shadow forecast qualification, not matched real continuation for every action candidate."),
        ("NanoWM", "reports/nanowm_security_readiness_decision_20261015.json",
         "Resources/native rollout inspected, but a complete fixed-candidate true-utility table is not present in the cited readiness evidence."),
        ("CartPole (mbrl-lib)", "runs/mbrl_rank_coupling_cartpole_v2/assessment.json",
         "Closed-loop conditions diverge after selection; recorded episode returns do not supply the counterfactual utilities of every candidate at one frozen root."),
        ("WorldAgen historical h30", "reports/worldagen_corrected_decision_evidence_audit_20261015.json",
         "Retired image/target/action protocol. Old selected-action results cannot be merged with corrected forecasts; only corrected three-step local utilities are admitted.")]
    result = []
    for system, source, reason in specs:
        doc = inputs.json(source)
        result.append({"system": system, "status": "excluded_from_numeric_bound", "reason": reason,
                       "source": source, "source_status": doc["status"], "leverage": None})
    return result


def csv_text(rows, fields=None):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def table_text(aggregates):
    choices = [("Cosmos", "success", "Success count"), ("Cosmos", "capped_steps", "Capped steps"),
               ("DINO-WM", "summed_native_reward_h30", "30-step return"),
               ("DINO-WM", "terminal_native_reward_h30", "Terminal reward"),
               ("DINO-WM", "ever_success_h30", "30-step success"),
               ("WorldAgen", "success_h3", "3-step success"),
               ("WorldAgen", "reward_sum_h3", "3-step reward")]
    lines = ["% Generated by scripts/compute_decision_leverage_panel.py; sums over fixed roots.",
             "\\begin{tabular}{llrrrr}", "\\hline",
             "System / interface & Utility & Roots & Range & Baseline & Harm cap \\\\", "\\hline"]
    for system, metric, label in choices:
        r = next(x for x in aggregates if x["system"] == system and x["metric"] == metric)
        fmt = lambda x: f"{x:.3f}".rstrip("0").rstrip(".")
        interval = f"[{fmt(r['minimum_total'])}, {fmt(r['maximum_total'])}]"
        lines.append(f"{system} & {label} & {r['roots']} & {interval} & "
                     f"{fmt(r['baseline_value_total'])} & {fmt(r['baseline_harm_headroom_total'])} \\\\")
    lines.extend(["\\hline", "\\end{tabular}",
                  "% DINO is an added fixed-bank selector, one root. WorldAgen is only a local utility diagnostic:",
                  "% three roots share one episode and no forecast-driven selector was run.",
                  "% Harm cap is baseline-specific; lower is better only for capped steps."])
    return "\n".join(lines) + "\n"


def build(root=ROOT):
    inputs = Inputs(root)
    rows, systems = [], []
    for adapter in (cosmos, dino, worldagen):
        values, info = adapter(inputs)
        rows.extend(values)
        systems.append(info)
    skipped = exclusions(inputs)
    roots, aggregate, clusters = summarize(rows)
    report = {"schema_version": 1, "status": "completed_cache_recomputation", "systems": systems,
              "excluded": skipped, "aggregates": aggregate, "root_bounds": roots, "cluster_bounds": clusters,
              "scope": "Finite recorded branches only. Utilities are environmental outcomes, not model scores. No random population claim.",
              "coverage": {"requested_system_families": 7, "numeric_families": 3,
                           "native_first_choice_family": 1, "added_fixed_bank_family": 1,
                           "local_utility_only_family": 1, "other_requested_families_without_qualified_bound": 4,
                           "not_exhaustive": "Explicitly audited sources, not a claim that every run directory is scientifically classified."},
              "proof_conditions": ["Original physical root, utility definition, candidate identities, and continuation rule remain fixed.",
                                   "Every selector output is an audited candidate identity; stochastic identity selection is a convex mixture.",
                                   "Elite-mean actions outside the finite bank have no certified utility bound from sampled utilities.",
                                   "Only recorded outcomes are bounded; no guarantee over new random environment realizations or future roots.",
                                   "Metric extrema are computed separately and need not be jointly attainable by a single selector.",
                                   "Baseline-specific harm capacity can be smaller than the total candidate utility span.",
                                   "Hindsight extrema are descriptive limits, not feasible attack policies."],
              "statistics": {"confidence_intervals": None, "p_values": None,
                             "reason": "Exact descriptive bounds on fixed development caches; correlated candidates/repetitions are not independent samples."},
              "source_bindings": sorted(inputs.bindings.values(), key=lambda r: r["path"]),
              "activity": {"model_calls": 0, "environment_steps": 0, "api_calls": 0, "checkpoint_loads": 0},
              "reproduction": "python scripts/compute_decision_leverage_panel.py --verify-only"}
    json_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return {"panel_results.json": json_text, "panel_candidates.csv": csv_text(rows),
            "panel_root_bounds.csv": csv_text(roots), "panel_aggregate.csv": csv_text(aggregate),
            "panel_clusters.csv": csv_text(clusters), "panel_excluded.csv": csv_text(skipped),
            "panel_table.tex": table_text(aggregate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--verify-only", action="store_true", help="Recompute all sources and compare every artifact without writes")
    args = parser.parse_args()
    artifacts = build()
    if args.verify_only:
        for name, content in artifacts.items():
            require((args.out_dir / name).read_text() == content, "Generated artifact differs: " + name)
    else:
        require(not any((args.out_dir / name).exists() for name in artifacts),
                "Output exists; use --verify-only or a fresh --out-dir")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            with (args.out_dir / name).open("x") as handle:
                handle.write(content)
    report = json.loads(artifacts["panel_results.json"])
    print(json.dumps({"status": "verified_exact" if args.verify_only else "written", "artifacts": list(artifacts),
                      "qualified_families": report["coverage"]["numeric_families"],
                      "audited_inputs": len(report["source_bindings"]), "model_calls": 0, "environment_steps": 0}))


if __name__ == "__main__":
    main()
