#!/usr/bin/env python3
"""Recompute two development DINO banks from tensors and real replay traces.

CPU tensor reductions only. No checkpoint, model, encoder, simulator, API,
training, or CEM is run. These heterogeneous development banks are not a
prospective independent evaluation panel; no pooled CI or success rate is made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/decision_leverage_paper/dino_cached_decomposition.json"
FEATURES = ("visual", "proprio")


def compute():
    import numpy as np
    import torch

    torch.set_num_threads(4)
    sources = {}

    def path(name):
        p = ROOT / name
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def read(name):
        return json.loads(path(name).read_text())

    def arrays(name):
        with np.load(path(name), allow_pickle=False) as a:
            return {k: a[k].copy() for k in a.files}

    def traces(name):
        rows = [json.loads(x) for x in path(name).read_text().splitlines()]
        assert len(rows) == 30
        assert all(np.isfinite(x["reward"]) for x in rows)
        return rows

    def utility(rows):
        return dict(terminal_reward=rows[-1]["reward"],
                    cumulative_reward=sum(x["reward"] for x in rows),
                    negative_terminal_distance=-rows[-1]["reach_distance"],
                    terminal_success=int(rows[-1]["success"]),
                    ever_success=int(any(x["success"] for x in rows)),
                    first_success_control=next((i + 1 for i, x in enumerate(rows) if x["success"]), None))

    def score(prediction, goal):
        terms = {k: (torch.from_numpy(prediction[k]) - torch.from_numpy(goal[k])).pow(2)
                 .mean(dim=tuple(range(2, prediction[k].ndim)))[-1] for k in FEATURES}
        return (terms["visual"] + .1 * terms["proprio"]).tolist()

    def error(prediction, truth):
        # Compare forecast to actual future, never future to goal. Double
        # precision reductions are common across the two cache conventions.
        terms = {k: np.mean((prediction[k][-1].astype(np.float64)
                            - truth[k][-1].astype(np.float64)) ** 2,
                           axis=tuple(range(1, prediction[k].ndim - 1))) for k in FEATURES}
        return {**{k: v.tolist() for k, v in terms.items()},
                "weighted": (terms["visual"] + .1 * terms["proprio"]).tolist()}

    def bank(bank_id, seed, names, truth, goal, forecasts, utilities, old_costs, old_truth, old_errors):
        def scored(prediction):
            # Preserve each historical reduction batch shape, since FP32
            # kernels may round differently after concatenating B1 into B2.
            if len(names) == 2:
                return [score({k: v[:, i:i + 1].copy() for k, v in prediction.items()}, goal)[0] for i in range(2)]
            return score(prediction, goal)
        true_scores = scored(truth)
        assert true_scores == old_truth
        n = len(names)
        assert len(true_scores) == n
        oracle_proxy = min(range(n), key=lambda i: (true_scores[i], i))
        conditions = {}
        for arm, pred in forecasts.items():
            scores, errs = scored(pred), error(pred, truth)
            assert scores == old_costs[arm], (bank_id, arm, scores, old_costs[arm])
            assert np.allclose(errs["weighted"], old_errors[arm], rtol=1e-6, atol=1e-7)
            selected = min(range(n), key=lambda i: (scores[i], i))
            metric_rows = {}
            for metric in ("terminal_reward", "cumulative_reward", "negative_terminal_distance", "terminal_success", "ever_success"):
                u = [x[metric] for x in utilities]
                best = max(u)
                mis, pred_delta = best - u[oracle_proxy], u[oracle_proxy] - u[selected]
                regret = best - u[selected]
                assert abs((mis + pred_delta) - regret) <= 1e-12
                metric_rows[metric] = dict(Lev=max(u) - min(u), Mis=mis, Pred=pred_delta,
                                          regret=regret, selected_utility=u[selected],
                                          true_future_proxy_selected_utility=u[oracle_proxy],
                                          best_recorded_utility=best,
                                          worst_recorded_utility=min(u))
            conditions[arm] = dict(costs=scores, ranking=sorted(range(n), key=lambda i: (scores[i], i)),
                                   selected=selected, terminal_forecast_mse=errs, metrics=metric_rows)
        contrasts = {}
        for arm, result in conditions.items():
            if arm == "original":
                continue
            base = conditions["original"]
            errs = [x - y for x, y in zip(result["terminal_forecast_mse"]["weighted"], base["terminal_forecast_mse"]["weighted"])]
            deltas = {k: result["metrics"][k]["selected_utility"] - base["metrics"][k]["selected_utility"] for k in result["metrics"]}
            contrasts[arm] = dict(terminal_forecast_error_delta_per_candidate=errs,
                                  mean_terminal_forecast_error_delta=float(np.mean(errs)),
                                  every_candidate_error_nonincreasing=all(x <= 0 for x in errs),
                                  at_least_one_candidate_error_strictly_lower=any(x < 0 for x in errs),
                                  selected_action_changed=result["selected"] != base["selected"],
                                  selected_real_utility_delta= deltas,
                                  fidelity_improves_and_terminal_reward_declines=(float(np.mean(errs)) < 0 and deltas["terminal_reward"] < 0))
        return dict(bank_id=bank_id, physical_seed=seed, independent_development_roots=1,
                    candidate_names=names, action_commitment_controls=30, native_execution_prefix_controls=15,
                    native_full_cem=False, true_future_native_costs=true_scores,
                    true_future_proxy_selected=oracle_proxy, utilities=utilities,
                    conditions=conditions, paired_contrasts=contrasts)

    # Five-candidate bank: all physical replay at seed1, although three sequences
    # were extracted from three different historical closed-loop trajectories.
    old_dir = "runs/dino_action_saturation_gate_20261015"
    exp_dir = "runs/dino_action_saturation_exploratory_20261015"
    prior, exp = read(old_dir + "/summary.json"), read(exp_dir + "/summary.json")
    old_truth = read(exp_dir + "/oracle_objective_analysis.json")
    assert prior["status"] == "completed_cpu_fixed_bank_gate"
    assert exp["status"] == "completed_exploratory_cached_alias_forecasts"
    assert prior["snapshot"]["historical_source_prefix_verified"]
    setup1 = read("configs/dino_action_saturation_gate_20261015.json")
    assert setup1["root"]["directory"].endswith("seed1_clean")
    cached, predicted = arrays(old_dir + "/forecasts_and_encodings.npz"), arrays(exp_dir + "/condition_forecasts.npz")
    frozen = arrays(old_dir + "/frozen_bank.npz")
    raw = frozen["requested_physical"]
    assert np.array_equal(raw[0, ..., :3].clip(-1, 1), raw[1, ..., :3].clip(-1, 1))
    assert np.array_equal(raw[0, ..., :3].clip(-1, 1), raw[2, ..., :3].clip(-1, 1))
    assert np.array_equal(raw[0, ..., 3], raw[1, ..., 3]) and np.array_equal(raw[0, ..., 3], raw[2, ..., 3])
    rows5 = []
    for i in range(5):
        rows = traces(f"{old_dir}/candidate{i}/original/trace.jsonl")
        ground = arrays(f"{old_dir}/candidate{i}/ground_truth.npz")
        assert np.array_equal(ground["rewards"], [x["reward"] for x in rows])
        for arm in ("inflated_saturated_xyz", "canonical_saturated_xyz"):
            alias_rows = traces(f"{old_dir}/candidate{i}/{arm}/trace.jsonl")
            for a, b in zip(rows, alias_rows):
                for key in ("reward", "state", "pixels", "physical", "success", "reach_distance"):
                    assert a[key] == b[key]
        u = utility(rows)
        saved = prior["utilities"][i]["primary30"]
        assert u["terminal_reward"] == saved["terminal_native_reward"]
        assert u["cumulative_reward"] == saved["summed_native_reward"]
        rows5.append(u)
    five = bank("seed1_five_historical_prefixes", 1, prior["bank"]["candidate_names"],
                {k: cached["ground_truth_" + k] for k in FEATURES},
                {k: cached["goal_" + k] for k in FEATURES},
                {arm: {k: predicted[arm + "_" + k] for k in FEATURES} for arm in exp["conditions"]},
                rows5, {a: v["native_costs"] for a, v in exp["conditions"].items()},
                old_truth["true_future_native_objective"]["costs"],
                {a: v["forecast_errors"]["weighted"]["final_per_candidate"] for a, v in exp["conditions"].items()})
    five["selection_scope"] = "A new ranker on historical trajectory prefixes, including no-op and sign reversal; development root and known failed original Clean competence gate. Not a native CEM candidate/elite bank."

    # Two means: actual H30 outcomes were independently replayed, and their
    # historical first15 controls exactly matched their native traces.
    names = ["clean_identity_copy", "canonical_request"]
    native = "runs/dino_native_alias_pair_20261015"
    diag = "runs/dino_native_alias_forecast_gpu_20261015"
    replay = "runs/dino_first_query_full_horizon_20261015"
    truthdir = "runs/dino_first_query_terminal_truth_20261015"
    ref = read(truthdir + "/summary.json")
    rank = read("reports/dino_first_query_two_mean_ranking_20261015.json")
    replay_ref = read(replay + "/summary.json")
    assert ref["status"] == "completed_calibrated_two_mean_terminal_truth" and ref["calibration"]["all_passed"]
    assert read("configs/dino_native_alias_pair_20261015.json")["task"]["physical_seed"] == 4
    actions = arrays(replay + "/physical_actions.npz")
    truths, goals, predictions, rows2, native_cache = [], [], [], [], []
    for i, name in enumerate(names):
        assert replay_ref["candidates"][i]["historical_prefix_exact_steps"] == 15
        truths.append(arrays(f"{truthdir}/{name}_control30.npz"))
        goals.append(arrays(f"{native}/{name}/goal_encoding.npz"))
        predictions.append(arrays(f"{diag}/{name}_query1.npz"))
        native_cache.append(arrays(f"{native}/{name}/query1_forecast_cache.npz"))
        assert np.array_equal(actions["mean_original"][i], native_cache[-1]["mean_original"])
        rows = traces(f"{replay}/{name}/trace.jsonl")
        u = utility(rows)
        assert u["cumulative_reward"] == ref["physical_utilities"][name]["cumulative_native_reward"]
        assert u["terminal_reward"] == ref["physical_utilities"][name]["terminal_native_reward"]
        rows2.append(u)
    for k in FEATURES:
        assert np.array_equal(goals[0][k], goals[1][k])
        assert np.array_equal(native_cache[0]["context_" + k], native_cache[1]["context_" + k])
    two = bank("seed4_two_posthoc_returned_means", 4, names,
               {k: np.concatenate([t[k] for t in truths], axis=1) for k in FEATURES}, goals[0],
               {arm: {k: np.concatenate([t[arm + "_" + k] for t in predictions], axis=1) for k in FEATURES} for arm in ("original", "canonical")},
               rows2, {arm: scores for arm, scores in zip(rank["cost_matrix"]["rows"], rank["cost_matrix"]["values"])},
               ref["true_future_objective"]["costs"],
               {arm: [ref["terminal_forecast_errors"][name][arm]["weighted_mse"] for name in names] for arm in ("original", "canonical")})
    two["selection_scope"] = "Post-hoc two full returned CEM means from one root; the fixed-bank argmin is an added diagnostic, not native CEM. No population confirmation."
    assert not torch.cuda.is_initialized()
    return dict(schema_version=1, status="verified_cached_development_banks",
                numerical_conventions={"native_goal_cost": "CPU FP32 native terminal visual MSE + 0.1 * proprio MSE; exact to recorded costs",
                                       "forecast_error": "FP64 reduction of FP32 forecast minus actual future encoding; distinct from goal distance",
                                       "utility": "Python double sum of actual independent replay rewards",
                                       "selection": "argmin native score, exact ties choose lowest index"},
                banks=[five, two],
                pooling="No pooled estimator, CI, p-value or independent population rate: distinct development banks, candidate selection histories and runtime protocols.",
                claims={"local_paired_fidelity_utility_reversal": True,
                        "independent_confirmatory_p1": False,
                        "native_full_cem_oracle_result": False,
                        "success_rate_degradation": False},
                activity={"models": 0, "encoders": 0, "environment_steps": 0, "checkpoints": 0, "api_calls": 0, "cuda_initialized": False},
                source_sha256=sources)


def csv_rows(result):
    rows = []
    for b in result["banks"]:
        for arm, cond in b["conditions"].items():
            for i, name in enumerate(b["candidate_names"]):
                rows.append(dict(bank=b["bank_id"], physical_seed=b["physical_seed"], condition=arm,
                                 candidate_index=i, candidate=name, forecast_selected=i == cond["selected"],
                                 true_future_proxy_selected=i == b["true_future_proxy_selected"],
                                 native_forecast_goal_cost=cond["costs"][i], true_future_goal_cost=b["true_future_native_costs"][i],
                                 forecast_to_truth_visual_mse=cond["terminal_forecast_mse"]["visual"][i],
                                 forecast_to_truth_proprio_mse=cond["terminal_forecast_mse"]["proprio"][i],
                                 forecast_to_truth_weighted_mse=cond["terminal_forecast_mse"]["weighted"][i],
                                 **b["utilities"][i]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    result = compute()
    if args.verify_only:
        assert json.loads(args.out.read_text()) == result
        with args.out.with_suffix(".csv").open(newline="") as handle:
            actual = list(csv.DictReader(handle))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in csv_rows(result)]
        assert actual == expected
    else:
        assert not args.out.exists() and not args.out.with_suffix(".csv").exists()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        rows = csv_rows(result)
        with args.out.with_suffix(".csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "verified" if args.verify_only else "written", "result": str(args.out), "banks": len(result["banks"]), "activity": result["activity"]}))


if __name__ == "__main__":
    main()
