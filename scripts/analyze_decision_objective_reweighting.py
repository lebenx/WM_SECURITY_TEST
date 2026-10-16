#!/usr/bin/env python3
"""Audit nonnegative objective reweighting on the archived DINO two-mean bank.

This is a deterministic calculation on cached measured features and rewards,
not a learned defense, model inference, or independent experimental replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("reports/dino_first_query_native_objective_decomposition_20261015.json")
DESTINATION = Path("reports/decision_leverage_paper/objective_reweighting.json")


def compute() -> dict:
    payload = (ROOT / SOURCE).read_bytes()
    source = json.loads(payload)
    assert source["status"] == "completed_cached_native_objective_decomposition"
    rows = source["components"]
    assert [r["candidate"] for r in rows] == ["clean_identity_copy", "canonical_request"]
    for row in rows:
        assert row["fp32_sum_exact_to_previous_report"]
    terms = ["visual_mse", "proprio_embedding_mse"]
    differences = {key: rows[1][key] - rows[0][key] for key in terms}
    assert all(math.isfinite(v) and v < 0 for v in differences.values())
    rewards = [r["already_measured_physical_utility"] for r in rows]
    assert rewards[0]["controls"] == rewards[1]["controls"] == 30
    utility_gaps = {key: rewards[1][key] - rewards[0][key] for key in
                    ["cumulative_native_reward", "terminal_native_reward"]}
    assert all(v < 0 for v in utility_gaps.values())
    grid = []
    for alpha in [0.0, 0.0001, 0.1, 1.0, 10.0, 1000.0, 1000000.0]:
        costs = [r["visual_mse"] + alpha * r["proprio_embedding_mse"] for r in rows]
        assert costs[1] < costs[0]
        grid.append({"visual_weight": 1.0, "proprio_weight": alpha,
                     "costs_float64_from_stored_fp32_components": costs,
                     "selected_candidate_index": 1})
    return {
        "schema_version": 1,
        "status": "completed_cache_only_component_dominance_audit",
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "scope": "One post-hoc DINO two-returned-mean bank; same root as the archived H30 truth diagnostic.",
        "statistical_units": {"independent_roots": 1, "fixed_candidates": 2,
                              "new_roots": 0, "confidence_interval": None},
        "candidate_order": source["candidate_order"],
        "true_future_component_costs": [{"candidate": r["candidate"],
                                          **{key: r[key] for key in terms}} for r in rows],
        "candidate1_minus_candidate0_components": differences,
        "candidate1_minus_candidate0_real_utility": utility_gaps,
        "illustrative_weight_grid_not_a_fitted_hyperparameter_search": grid,
        "all_nonnegative_nonzero_weights_statement": {
            "equation": "C1-C0 = w_visual * delta_visual + w_proprio * delta_proprio < 0",
            "assumptions": "Real arithmetic; w_visual>=0, w_proprio>=0, at least one strictly positive; the archived feature components and candidate support are unchanged.",
            "conclusion": "Every such weighted sum prefers candidate1, the lower-return and lower-terminal-reward branch. Raising the proprio weight alone cannot correct this true-future ordering.",
            "degenerate_exception": "Both weights zero make all candidates tie; tie-breaking then supplies the choice, not objective calibration. Extreme finite-precision underflow/overflow is not a substantive repair.",
        },
        "limits": [
            "This concerns oracle-encoded future costs on this fixed bank, not every model-predicted cost or all states.",
            "It does not establish that all scorer-side changes fail; different features, rewards, or nonlinear scorers are outside this weight family.",
            "It is not a comparison of trained or deployed defenses, and does not test P3 on held-out roots.",
            "No weight is selected using test utility; the all-weights statement follows from both component differences being strictly negative.",
        ],
        "activity": {"model_calls": 0, "environment_steps": 0, "api_calls": 0,
                     "gpu_calls": 0, "training_updates": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / DESTINATION)
    args = parser.parse_args()
    result = compute()
    if args.verify_only:
        assert json.loads(args.out.read_text()) == result, "Saved audit differs from source recomputation"
        print("Verified component dominance, illustrative weights, utility gaps and source hash.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
