#!/usr/bin/env python3
"""Post hoc timing decomposition of existing DINO reach-v3/reward-v2 traces.

Standard library only; no simulator/model imports, new episodes, or hashes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dino_native_alias_pair_20261015"
OUT = ROOT / "reports/dino_native_alias_utility_timing_20261015.json"
ARMS = ("clean_identity_copy", "canonical_request")
REWARD_SOURCE = ROOT / ".venv-a2world/lib/python3.11/site-packages/metaworld/envs/sawyer_reach_v3.py"
RUNNER_SOURCE = ROOT / "scripts/pilot_dino_native_alias_pair.py"
TOLERANCE = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=TOLERANCE)


def describe(rows):
    n = len(rows)
    total = math.fsum(r["reward"] for r in rows)
    return {"controls": n, "first_policy_step": rows[0]["policy_step"] if n else None,
            "last_policy_step": rows[-1]["policy_step"] if n else None,
            "reward_sum": total, "mean_reward": total / n if n else None,
            "successful_controls": sum(r["success"] for r in rows),
            "mean_reach_distance": math.fsum(r["reach_distance"] for r in rows) / n if n else None}


def analyze():
    summary = json.loads((RUN / "summary.json").read_text())
    require(summary["status"] == "completed_native_alias_pair", "Pair not complete")
    require(summary["setup"]["elapsed_after_warmup"] == 1, "Different warmup scope")
    # Read source text, without importing the environment or rerunning its reward.
    reward_source = REWARD_SOURCE.read_text()
    runner_source = RUNNER_SOURCE.read_text()
    require('if self.reward_function_version == "v2"' in reward_source
            and "return (10 * in_place, tcp_to_target, in_place)" in reward_source
            and "success = float(reach_dist <= 0.05)" in reward_source,
            "Expected observed reach-v3 reward-v2 definition not found")
    require('raw.reward_function_version == "v2"' in runner_source,
            "Original runner reward-version guard not found")
    traces, totals, first_hits, per_arm = {}, {}, {}, {}
    for arm in ARMS:
        rows = [json.loads(line) for line in (RUN / arm / "trace.jsonl").read_text().splitlines()]
        record = summary["arms"][arm]
        outcome = record["outcome"]
        require(record["status"] == "completed", "Incomplete arm")
        require([r["policy_step"] for r in rows] == list(range(1, 100)), "Not the fixed99 controls")
        require([r["environment_step"] for r in rows] == list(range(2, 101)), "Warmup/clock mismatch")
        require(outcome["policy_controls_executed"] == 99 and outcome["total_environment_steps"] == 100,
                "Source step budget mismatch")
        require(all(type(r["success"]) is bool and math.isfinite(r["reward"])
                    and 0 <= r["reward"] <= 10 and math.isfinite(r["reach_distance"]) for r in rows),
                "Invalid reward or success record")
        require(all(r["success"] == (r["reach_distance"] <= .05) for r in rows),
                "Recorded success and physical distance threshold disagree")
        require(all(r["reward"] == 10 for r in rows if r["success"]),
                "Successful controls do not have the expected reward10")
        total = math.fsum(r["reward"] for r in rows)
        require(close(total, outcome["reward_sum"]), "True trace return disagrees with summary")
        require(rows[-1]["success"] == outcome["final_success"]
                and any(r["success"] for r in rows) == outcome["ever_success"], "Outcome mismatch")
        first = next((r["policy_step"] for r in rows if r["success"]), None)
        require(first is not None, "Both observed arms must have reached success")
        traces[arm], totals[arm], first_hits[arm] = rows, total, first
        per_arm[arm] = {
            "full_episode": describe(rows), "first_success_policy_step": first,
            "final_success": rows[-1]["success"], "ever_success": True,
            "terminal_reward": rows[-1]["reward"], "final_reach_distance": rows[-1]["reach_distance"],
            "success_region": describe([r for r in rows if r["success"]]),
            "non_success_region": describe([r for r in rows if not r["success"]]),
            "before_own_first_success": describe([r for r in rows if r["policy_step"] < first]),
            "from_own_first_success_inclusive": describe([r for r in rows if r["policy_step"] >= first]),
        }
    clean, canonical = (traces[a] for a in ARMS)
    total_delta = totals[ARMS[1]] - totals[ARMS[0]]
    cutoff = min(first_hits.values())

    def paired_window(start, end):
        arm_stats = {a: describe([r for r in traces[a] if start <= r["policy_step"] <= end]) for a in ARMS}
        return {"start_policy_step": start, "end_policy_step": end, "arms": arm_stats,
                "canonical_minus_clean_return": arm_stats[ARMS[1]]["reward_sum"] - arm_stats[ARMS[0]]["reward_sum"]}

    windows = {"before_either_first_success": paired_window(1, cutoff - 1),
               "from_earliest_first_success_inclusive": paired_window(cutoff, 99)}
    region_deltas = {region: per_arm[ARMS[1]][region]["reward_sum"] - per_arm[ARMS[0]][region]["reward_sum"]
                     for region in ("success_region", "non_success_region")}
    pointwise = [{"policy_step": a["policy_step"], "environment_step": a["environment_step"],
                  "canonical_minus_clean_reward": b["reward"] - a["reward"],
                  "clean_reward": a["reward"], "canonical_reward": b["reward"],
                  "clean_success": a["success"], "canonical_success": b["success"],
                  "clean_reach_distance": a["reach_distance"], "canonical_reach_distance": b["reach_distance"],
                  "clean_planning_call": a.get("planning_call"), "canonical_planning_call": b.get("planning_call")}
                 for a, b in zip(clean, canonical)]
    minimum = min(r["canonical_minus_clean_reward"] for r in pointwise)
    worst = [r for r in pointwise if r["canonical_minus_clean_reward"] == minimum]
    stages = []
    if all("planning_call" in r for rows in traces.values() for r in rows):
        keys = sorted({(a["planning_call"], b["planning_call"]) for a, b in zip(clean, canonical)})
        for ca, cb in keys:
            indices = [i for i, (a, b) in enumerate(zip(clean, canonical))
                       if (a["planning_call"], b["planning_call"]) == (ca, cb)]
            stats = {arm: describe([traces[arm][i] for i in indices]) for arm in ARMS}
            stages.append({"clean_planning_call": ca, "canonical_planning_call": cb,
                           "first_policy_step": indices[0] + 1, "last_policy_step": indices[-1] + 1,
                           "arms": stats,
                           "canonical_minus_clean_return": stats[ARMS[1]]["reward_sum"] - stats[ARMS[0]]["reward_sum"]})
    checks = {
        "common_window_deltas_sum_to_full_delta": close(math.fsum(w["canonical_minus_clean_return"] for w in windows.values()), total_delta),
        "region_deltas_sum_to_full_delta": close(math.fsum(region_deltas.values()), total_delta),
        "pointwise_deltas_sum_to_full_delta": close(math.fsum(r["canonical_minus_clean_reward"] for r in pointwise), total_delta),
        "planning_stage_deltas_sum_to_full_delta": close(math.fsum(s["canonical_minus_clean_return"] for s in stages), total_delta) if stages else None,
    }
    for arm in ARMS:
        d = per_arm[arm]
        checks[f"{arm}_region_sum"] = close(d["success_region"]["reward_sum"] + d["non_success_region"]["reward_sum"], totals[arm])
        checks[f"{arm}_own_hit_segment_sum"] = close(d["before_own_first_success"]["reward_sum"] + d["from_own_first_success_inclusive"]["reward_sum"], totals[arm])
    require(all(v is True for v in checks.values() if v is not None), "Decomposition identity failed")
    return {
        "schema_version": 1, "status": "completed_existing_trace_utility_timing_decomposition",
        "source_paths": [str((RUN / "summary.json").relative_to(ROOT)),
                         *[str((RUN / a / "trace.jsonl").relative_to(ROOT)) for a in ARMS],
                         str(REWARD_SOURCE.relative_to(ROOT)), str(RUNNER_SOURCE.relative_to(ROOT))],
        "environment_reward_contract": {
            "implementation": "Observed MetaWorld sawyer_reach_v3.py with reward_function_version=v2, as required by the original runner.",
            "reward": "10*tolerance(TCP-target distance, bounds=(0,.05), margin=norm(hand_init_pos-target), sigmoid=long_tail)",
            "success": "TCP-target distance <=.05; every recorded successful control has reward exactly10.",
            "fixed_exposure": "Both arms have99 policy controls plus the same one-control warmup; no success-based truncation. All decomposition sums exclude that shared warmup.",
            "source_scope": "Source text and recorded trace checks only; no environment import, fresh reward recomputation, or source-file hash audit."},
        "per_arm": per_arm,
        "full_return_effect": {"clean_return": totals[ARMS[0]], "canonical_return": totals[ARMS[1]],
                               "canonical_minus_clean_return": total_delta,
                               "relative_return_drop": -total_delta / totals[ARMS[0]],
                               "shared_warmup_reward_excluded": summary["setup"]["warmup_reward_excluded_from_policy_return"]},
        "common_clock_windows": {"rule": "1..min(first_success_policy_step)-1 and the remaining controls. Defined from observed hit times, not optimized to maximize the return contrast.",
                                 "earliest_first_success_policy_step": cutoff, "windows": windows},
        "success_region_delta_decomposition": region_deltas,
        "single_step_largest_loss": {"minimum_canonical_minus_clean_reward": minimum,
                                     "ties": len(worst), "tie_rule": "Report every exact minimum in policy-step order.",
                                     "records": worst},
        "planning_stage_groups": stages,
        "identity_checks": {"absolute_tolerance": TOLERANCE, "all_passed": True, "checks": checks},
        "interpretation": [
            "The recorded native dense-return loss remains valid; fixed exposure rules out unequal episode lengths as its source.",
            "The common-clock decomposition localizes the observed net loss to the interval before either arm first succeeds; the remaining interval has a small positive canonical-minus-clean return.",
            "Canonicalization gains successful-region reward through more successful controls, while losing more reward outside the success region. This is not reduced successful occupancy.",
            "Earlier first attainment, successful-control count and final distance are distinct utilities from integrated dense return; neither set should be suppressed or substituted for the registered primary endpoint."],
        "limits": [
            "Post hoc descriptive decomposition of one development physical root. The99 steps and7 planning stages are correlated observations, not independent experimental samples; no significance test or confidence interval.",
            "Own-first-success segments and success/non-success subsets are endogenous to each arm and have unequal membership/length. Their differences are accounting identities, not causal effects within comparable state populations.",
            "Common-clock contrasts also compare downstream trajectories with different states/actions. Timing localization is not identification of a causal mediator or a same-state candidate-ranking effect.",
            "No prediction-error mediation, full-horizon objective attribution, goal-attainment harm, generalization or novel defense is established. Existing return/recovery evidence and paper-level No-Go are unchanged.",
            "Successful occupancy is reported as discrete observed control counts, not independently measured continuous dwell time."],
        "activity": {"model_calls": 0, "environment_steps": 0, "api_requests": 0, "new_trajectories": 0, "hashes_recomputed": 0},
        "reproduction": f"{sys.executable} scripts/analyze_dino_native_alias_utility_timing.py --verify-only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(args.verify_only or not args.out.exists(), "Refusing to overwrite report")
    report = analyze()
    if not args.verify_only:
        with args.out.open("x") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": report["status"], "return_effect": report["full_return_effect"],
                      "common_window_deltas": {k: v["canonical_minus_clean_return"] for k, v in report["common_clock_windows"]["windows"].items()},
                      "success_region_deltas": report["success_region_delta_decomposition"],
                      "largest_loss": report["single_step_largest_loss"],
                      "stage_count": len(report["planning_stage_groups"]), "identities_passed": True,
                      "files_written": [] if args.verify_only else [str(args.out)]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
