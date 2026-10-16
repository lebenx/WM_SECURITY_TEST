#!/usr/bin/env python3
"""Recompute cached H30 native-objective terms; no models, environments or APIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/dino_first_query_native_objective_decomposition_20261015.json"
NAMES = ("clean_identity_copy", "canonical_request")


def recompute():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    def load(path):
        with np.load(ROOT / path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
    prior = json.loads((ROOT / "reports/dino_first_query_terminal_truth_20261015.json").read_text())
    ranking = json.loads((ROOT / "reports/dino_first_query_two_mean_ranking_20261015.json").read_text())
    assert prior["status"] == "completed_calibrated_two_mean_terminal_truth"
    assert ranking["status"] == "completed_cached_two_mean_ranking"
    goal = load("runs/dino_native_alias_pair_20261015/clean_identity_copy/goal_encoding.npz")
    components = []
    for index, name in enumerate(NAMES):
        truth = load(f"runs/dino_first_query_terminal_truth_20261015/{name}_control30.npz")
        terms = {}
        for key in ("visual", "proprio"):
            target, obs = torch.from_numpy(goal[key]), torch.from_numpy(truth[key])
            assert target.dtype == obs.dtype == torch.float32 and target.shape == obs.shape
            terms[key] = (target - obs).pow(2).mean(dim=tuple(range(2, obs.ndim)))[-1]
        weighted = .1 * terms["proprio"]
        total = terms["visual"] + weighted
        expected = prior["true_future_objective"]["costs"][index]
        assert float(total.item()) == expected, "Native FP32 component sum differs from prior objective"
        raw = load(f"runs/dino_first_query_full_horizon_20261015/{name}/endpoint30.npz")
        components.append({"candidate": name, "visual_mse": float(terms["visual"].item()),
            "proprio_embedding_mse": float(terms["proprio"].item()), "weighted_proprio_term": float(weighted.item()),
            "native_weighted_cost_fp32": float(total.item()), "previous_native_cost": expected,
            "fp32_sum_exact_to_previous_report": True, "cached_raw_proprio": raw["proprio"].tolist(),
            "already_measured_physical_utility": prior["physical_utilities"][name]})
    true_gap = components[1]["native_weighted_cost_fp32"] - components[0]["native_weighted_cost_fp32"]
    gaps = {}
    for row, representation in zip(ranking["cost_matrix"]["values"], ranking["cost_matrix"]["rows"]):
        gap = row[1] - row[0]
        gaps[representation] = {"candidate1_minus_candidate0_cost": gap,
            "minus_true_future_cost_gap": gap - true_gap,
            "selected_candidate": ranking["rankings"][representation]["selected_candidate"]}
    assert not torch.cuda.is_initialized()
    return {"components": components,
        "component_candidate1_minus_candidate0": {key: components[1][key] - components[0][key] for key in
            ("visual_mse", "proprio_embedding_mse", "weighted_proprio_term", "native_weighted_cost_fp32")},
        "existing_score_gap_identity": {
            "definition": "For each representation, predicted gap = true-future proxy gap + gap error; gap is candidate1 cost minus candidate0 cost, not physical reward. These are identities on already reported scores.",
            "true_future_cost_gap": true_gap, "representations": gaps,
            "interpretation": "Original forecasts reverse the native true-future proxy preference, while that selected candidate happens to have higher measured30-control physical reward. Canonical forecasts recover the true-future proxy ordering more closely on this bank, and select the candidate with lower measured reward. This is not a causal mediation percentage and does not exclude all roles of forecast error in the full planner."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true", help="Compare every computed field to the existing report without writing")
    mode.add_argument("--out", type=Path, help="Write the recomputed numeric section to a new JSON, never overwrite")
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise FileExistsError(args.out)
    calculated = recompute()
    if args.verify_only:
        saved = json.loads(REPORT.read_text())
        for key, value in calculated.items():
            assert saved[key] == value, "Cached report differs: " + key
        print(json.dumps({"status": "verified_exact_numeric_sections", "report": str(REPORT),
                          "neural_forwards": 0, "environment_steps": 0, "api_calls": 0}))
    else:
        result = {"status": "completed_cached_native_objective_decomposition_numeric_reproduction",
            "scope": "Only two post-hoc selected H30 means and original goal; no full-CEM mediation or weight tuning.",
            "semantic_source_report": str(REPORT.relative_to(ROOT)), **calculated}
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": result["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
