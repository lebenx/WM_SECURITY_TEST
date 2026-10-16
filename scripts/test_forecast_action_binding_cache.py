#!/usr/bin/env python3
"""R001: hypothetical response/action association fault, cached arithmetic only.

SIMULATED MAPPING FAULT, NOT A VERIFIED NATIVE ATTACK SURFACE. These three
heterogeneous cache diagnostics are NOT three-system security validation.
No model loads, tensor experiments, environment runs, hash re-audits or APIs.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/forecast_action_binding_cache.json"
ARMS = ("identity", "reverse_without_ids", "reverse_with_ids_restored")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def permutation_check(order, n):
    require(sorted(order) == list(range(n)), "Not a candidate permutation")


def ranked_choice(scores, order, restore_ids):
    """order[position] is original response candidate ID, NOT its inverse."""
    permutation_check(order, len(scores))
    require(all(math.isfinite(x) for x in scores), "Nonfinite cached score")
    received = [scores[candidate_id] for candidate_id in order]
    if restore_ids:
        restored = [None] * len(scores)
        for candidate_id, value in zip(order, received):
            restored[candidate_id] = value
        costs = restored
    else:
        costs = received
    ranking = sorted(range(len(scores)), key=lambda i: (costs[i], i))
    return ranking[0], ranking, costs


def index_projection(native_selected, order, restore_ids):
    """Winner identity projection only; makes no claim to reconstruct score ties."""
    permutation_check(order, len(order))
    position = order.index(native_selected)
    return order[position] if restore_ids else position


def self_check():
    # Non-involutive cycle detects using order[winner] instead of inverse(order).
    require(index_projection(0, [1, 2, 0], False) == 2, "Inverse mapping error")
    for scores, perm in [([0., 0., 2.], [1, 2, 0]), ([4., 1., 1.], [2, 0, 1]), ([2., 4., 0.], [1, 2, 0])]:
        baseline = ranked_choice(scores, list(range(len(scores))), False)
        restored = ranked_choice(scores, perm, True)
        require(restored == baseline, "ID repair must also preserve exact tie-breaking")
    require(ranked_choice([0., 0., 2.], [2, 1, 0], False)[0] == 1, "Positional tie rule differs")
    try:
        permutation_check([0, 0], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate candidate IDs were accepted")


def read(path):
    return json.loads((ROOT / path).read_text())


def record(system, root, utilities, directions, choices, rankings, scores, source, mode):
    original = choices["identity"]
    require(choices["reverse_with_ids_restored"] == original, "ID restoration did not restore decision")
    rows = []
    for arm in ARMS:
        selected = choices[arm]
        for metric, higher in directions.items():
            values = [u[metric] for u in utilities]
            absolute, baseline = values[selected], values[original]
            delta = absolute - baseline
            rows.append(dict(system=system, root=root, condition=arm, mode=mode,
                             selected_candidate=selected, original_selected_candidate=original,
                             selected_action_flip=selected != original,
                             ranking_flip=None if rankings[arm] is None else rankings[arm] != rankings["identity"],
                             metric=metric, higher_is_better=higher, selected_real_utility=absolute,
                             original_real_utility=baseline, delta_vs_identity=delta,
                             harm_oriented_delta=-delta if higher else delta,
                             candidate_utility_span=max(values) - min(values), utility_source=source))
    return {"system": system, "root": root, "mode": mode, "fixed_permutation": list(reversed(range(len(utilities)))),
            "candidate_utilities": utilities, "choices": choices, "rankings": rankings,
            "cached_scores_by_condition": scores, "source": source}, rows


def build():
    self_check()
    records, rows, sources = [], [], []
    run = "runs/decision_leverage_dino_cpu_confirmatory"
    receipt = read(run + "/summary.json")
    require(receipt["status"] == "completed" and receipt["completed_seeds"] == list(range(1001, 1031)), "DINO30 panel incomplete")
    sources.append(run + "/summary.json")
    for seed in range(1001, 1031):
        source = f"{run}/seed{seed}/result.json"
        d = read(source)
        require(d["status"] == "completed" and d["seed"] == seed, "DINO root incomplete")
        costs = d["conditions"]["0.0"]["native_costs"]
        require(len(costs) == 5, "DINO bank size changed")
        choices, rankings, scores = {}, {}, {}
        for arm in ARMS:
            order = list(range(5)) if arm == "identity" else list(reversed(range(5)))
            choices[arm], rankings[arm], scores[arm] = ranked_choice(costs, order, arm == "reverse_with_ids_restored")
        require(choices["identity"] == d["conditions"]["0.0"]["selected_candidate"], "DINO recorded selector differs")
        utility = [u["h30"] for u in d["candidate_utilities"]]
        directions = {"terminal_reward": True, "cumulative_reward": True, "terminal_success": True, "ever_success": True}
        entry, lines = record("DINO", f"seed{seed}", utility, directions, choices, rankings, scores, source,
                              "hypothetical_positional_scored_response_mapping_on_added_fixed_bank")
        records.append(entry); rows.extend(lines); sources.append(source)

    source = "runs/cosmos_first_candidate_utility_recovery_analysis_v1/analysis.json"
    cosmos = read(source); sources.append(source)
    require(len(cosmos["pairs"]) == 30 and cosmos["validated_branches"] == 60 and cosmos["complete_amended_grid"], "Cosmos complete bank absent")
    require(len({(x["task_id"], x["init_id"]) for x in cosmos["pairs"]}) == 30, "Duplicate Cosmos root")
    for pair in cosmos["pairs"]:
        chosen = pair["source_native_selected_index"]
        require(chosen in (0, 1), "Unexpected Cosmos native selection")
        utility = [{"success": int(pair[f"candidate{i}_success"]), "capped_steps": pair[f"candidate{i}_capped_steps"]} for i in (0, 1)]
        choices = {arm: index_projection(chosen, [0, 1] if arm == "identity" else [1, 0], arm == "reverse_with_ids_restored") for arm in ARMS}
        entry, lines = record("Cosmos", f"task{pair['task_id']:02d}_init{pair['init_id']:02d}", utility,
                              {"success": True, "capped_steps": False}, choices, dict.fromkeys(ARMS), dict.fromkeys(ARMS), source,
                              "native_winner_index_projection_only_no_score_or_tensor_reranking")
        records.append(entry); rows.extend(lines)

    source = "runs/worldagen_prediction_contract_cpu_v2/summary.json"
    world = read(source); sources.append(source)
    require(world["status"] == "completed_prediction_contract_gate" and world["forecast_horizon"] == 3
            and world["policy_output_consumed"] is False and len(world["rows"]) == 3, "WorldAgen bypass contract changed")
    for root in world["rows"]:
        utility = []
        for i in (0, 1):
            repeats = sorted([x for x in root["replays"] if x["candidate"] == i], key=lambda x: x["repetition"])
            require(len(repeats) == 2 and all(x["initial_exact"] and x["future_exact"] for x in repeats), "WorldAgen replay mismatch")
            require(repeats[0]["reward_sum"] == repeats[1]["reward_sum"] and repeats[0]["success_after_each_action"] == repeats[1]["success_after_each_action"], "WorldAgen repeat utility differs")
            utility.append({"reward_sum_h3": repeats[0]["reward_sum"], "success_h3": int(any(repeats[0]["success_after_each_action"]))})
        # Forecast outputs are downstream of fixed native actions. A mapping
        # change cannot be promoted into a nonexistent forecast-driven selector.
        entry, lines = record("WorldAgen", f"chunk{root['chunk_index']:03d}", utility,
                              {"reward_sum_h3": True, "success_h3": True}, dict.fromkeys(ARMS, 0),
                              dict.fromkeys(ARMS), dict.fromkeys(ARMS), source,
                              "native_forecast_bypass_constant_candidate0_topology_and_local_span_only")
        records.append(entry); rows.extend(lines)

    aggregates = []
    for system in ("DINO", "Cosmos", "WorldAgen"):
        metrics = sorted({r["metric"] for r in rows if r["system"] == system})
        for metric in metrics:
            for arm in ARMS:
                part = [r for r in rows if (r["system"], r["metric"], r["condition"]) == (system, metric, arm)]
                n = len(part)
                aggregates.append({"system": system, "metric": metric, "condition": arm, "roots": n,
                    "sum_real_utility": sum(x["selected_real_utility"] for x in part),
                    "mean_real_utility": sum(x["selected_real_utility"] for x in part) / n,
                    "mean_delta_vs_identity": sum(x["delta_vs_identity"] for x in part) / n,
                    "sum_delta_vs_identity": sum(x["delta_vs_identity"] for x in part),
                    "harm_roots": sum(x["harm_oriented_delta"] > 0 for x in part),
                    "tie_roots": sum(x["harm_oriented_delta"] == 0 for x in part),
                    "improved_roots": sum(x["harm_oriented_delta"] < 0 for x in part),
                    "selected_action_flip_count": sum(x["selected_action_flip"] for x in part),
                    "selected_action_flip_rate": sum(x["selected_action_flip"] for x in part) / n,
                    "ranking_flip_count": None if part[0]["ranking_flip"] is None else sum(x["ranking_flip"] for x in part),
                    "identity_restoration_exact": all(x["delta_vs_identity"] == 0 and not x["selected_action_flip"] for x in part) if arm == "reverse_with_ids_restored" else None})
    require(all(x["identity_restoration_exact"] for x in aggregates if x["condition"] == "reverse_with_ids_restored"), "ID control failed")
    return {"schema_version": 1, "status": "completed_descriptive_cache_mapping_test", "registration": "IDEAS.md:R001",
            "warning": "SIMULATED MAPPING FAULT IS NOT A VERIFIED NATIVE ATTACK SURFACE. NOT THREE-SYSTEM SECURITY VALIDATION.",
            "operation": "One preregistered reverse permutation; action list fixed. No root selection, utility-directed permutation, new forecasts or physical replays.",
            "tie_rule": "DINO stable minimum score then fixed action index; ID restoration restores original candidate IDs before this tie rule. Cosmos has no scored reranking in this test; its original winner ID is only projected to the reversed position, so unknown score ties/native tie resolution are not reconstructed.",
            "scope_by_system": {"DINO": "30 fresh roots, added five-action H30 ranker. Cached original forecast scores are permuted; no forecast tensor or native CEM is executed.",
                                "Cosmos": "30 task/init development roots, ten task clusters; two real first-action continuations each. Native selected-index projection ONLY; no score availability or tensor-level permutation is claimed.",
                                "WorldAgen": "Three points on one trajectory, native forecast bypass. Keep candidate0 selected in every arm. Three-control zero-span/topology diagnostic, not an attack trial."},
            "aggregates": aggregates, "roots": records, "sources": sources,
            "statistics": "Descriptive absolute utilities and paired root differences only. No new bootstrap, p-value, population ASR or cross-system aggregation.",
            "id_control_limit": "Exact restoration follows from bookkeeping in this simulated association test. It is not a validated defense at an established external boundary.",
            "activity": {"model_loads": 0, "model_calls": 0, "tensor_experiments": 0, "environment_steps": 0,
                         "api_calls": 0, "gpu_calls": 0, "hash_reaudits": 0},
            "functional_checks": ["non-involutive cycle uses inverse mapping", "exact ties restore original ID tie-breaking", "invalid duplicate ID permutation rejected"]}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    require(not args.out.exists() and not args.out.with_suffix(".csv").exists(), "Refusing existing outputs")
    result, rows = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.out.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": result["status"], "warning": result["warning"], "reverse_results": [x for x in result["aggregates"] if x["condition"] == "reverse_without_ids"], "id_restoration_exact": True}))


if __name__ == "__main__":
    main()
