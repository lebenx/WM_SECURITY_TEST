"""Offline, descriptive audit of an already completed fixed coupling pair.

No model, simulator, optimizer, or API is loaded. Same-query counterfactuals
reuse the actually visited attack states, candidate actions, and forecast banks.
They are not counterfactual closed-loop episodes or independent observations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/mbrl_action_mean_coupling_20261015_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / args.run
    out = run / "analysis.json"
    table = run / "same_query_counterfactuals.csv"
    if out.exists() or table.exists():
        raise FileExistsError("Refusing to overwrite offline analysis")
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "completed":
        raise RuntimeError("A completed paired experiment is required")
    cfg = summary["configuration"]
    torch.set_num_threads(1)
    k, population = cfg["bank_size"], cfg["population"]
    elite_num = int(np.ceil(population * cfg["elite_ratio"]))
    alpha = cfg["alpha"]
    arm = run / "positive_action_mean"
    episode = json.loads((arm / "episode.json").read_text())
    records, trajectory = episode["operation_records"], episode["trajectory"]
    assert len(records) == len(trajectory) == episode["summary"]["steps"]
    rows = []
    for step, (record, transition) in enumerate(zip(records, trajectory)):
        with np.load(arm / f"final_query_{step:03d}.npz") as saved:
            data = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
        bank, actions, actual = data["bank"], data["actions"], data["scores"]
        ordered, original_indices = bank.sort(dim=0)
        all_rows = (torch.arange(k)[:, None] + data["phases"][None]) % k
        selected = ordered.gather(0, all_rows)
        selected_original = original_indices.gather(0, all_rows)
        multiset_exact = torch.equal(selected.sort(dim=0).values, ordered)
        index_bijection = torch.equal(selected_original.sort(dim=0).values,
                                      torch.arange(k)[:, None].expand(k, population))
        assert multiset_exact and index_bijection
        assert torch.equal(data["indices"], (record["fresh_shared_row"] + data["phases"]) % k)
        assert torch.equal(actual, ordered[data["indices"], torch.arange(population)])
        independent = ordered[data["independent_indices"], torch.arange(population)]
        actual_elites = actual.topk(elite_num).indices
        independent_elites = independent.topk(elite_num).indices
        mu = torch.tensor(record["mu_before_final_first"], dtype=actions.dtype)
        actual_action = alpha * mu + (1 - alpha) * actions[actual_elites].mean(dim=0)[0, 0]
        independent_action = alpha * mu + (1 - alpha) * actions[independent_elites].mean(dim=0)[0, 0]
        assert float(actual_action) == record["returned_action"] == transition["action"][0]
        before = float(actions[:, 0, 0][ordered.topk(elite_num, dim=-1).indices].mean(-1).mean())
        after = float(actions[:, 0, 0][selected.topk(elite_num, dim=-1).indices].mean(-1).mean())
        assert before == record["phase_info"]["before"]
        assert after == record["phase_info"]["after"]
        assert record["phase_info"]["scalar_column_multisets_exact"]
        assert record["phase_info"]["original_bank_indices_bijective"]
        actual_top_count = int((actual == actual.max()).sum())
        independent_top_count = int((independent == independent.max()).sum())
        actual_boundary_count = int((actual == actual[actual_elites[-1]]).sum())
        independent_boundary_count = int((independent == independent[independent_elites[-1]]).sum())
        rows.append({"step": step, "native_topk35_first_index_changed": int(actual_elites[0]) != int(independent_elites[0]),
            "elite_set_changed": set(actual_elites.tolist()) != set(independent_elites.tolist()),
            "first_action_changed_exact": float(actual_action) != float(independent_action),
            "first_action_changed_gt_1e_6": abs(float(actual_action - independent_action)) > 1e-6,
            "actual_first_action": float(actual_action), "same_query_independent_first_action": float(independent_action),
            "signed_action_difference": float(actual_action - independent_action),
            "absolute_action_difference": float((actual_action - independent_action).abs()),
            "phase_objective_before": before, "phase_objective_after": after,
            "phase_objective_improved_gt_tolerance": after > before + cfg["phase_improvement_tolerance"],
            "actual_top_score_tie_count": actual_top_count, "independent_top_score_tie_count": independent_top_count,
            "actual_elite_boundary_tie_count": actual_boundary_count,
            "independent_elite_boundary_tie_count": independent_boundary_count,
            "scalar_multisets_exact": multiset_exact, "original_bank_index_bijection": index_bijection})
    rates = {key: {"count": sum(int(row[key]) for row in rows),
                   "rate": sum(int(row[key]) for row in rows) / len(rows)}
             for key in ["native_topk35_first_index_changed", "elite_set_changed", "first_action_changed_exact",
                         "first_action_changed_gt_1e_6", "phase_objective_improved_gt_tolerance",
                         "scalar_multisets_exact", "original_bank_index_bijection"]}
    unique_both = [r for r in rows if r["actual_top_score_tie_count"] == r["independent_top_score_tie_count"] == 1]
    actual_returns = {}
    for name in cfg["arms"]:
        source = json.loads((run / name / "episode.json").read_text())
        tr = source["trajectory"]
        assert all(tr[i]["next_obs"] == tr[i + 1]["obs"] for i in range(len(tr) - 1))
        total = sum(t["reward"] for t in tr)
        assert total == source["summary"]["real_return"]
        actual_returns[name] = {"recorded_reward_sum": total, "steps": len(tr),
                               "terminated": tr[-1]["terminated"], "truncated": tr[-1]["truncated"]}
    action_delta = np.asarray([r["signed_action_difference"] for r in rows])
    gains = np.asarray([r["phase_objective_after"] - r["phase_objective_before"] for r in rows])
    result = {"status": "completed_offline_same_query_audit", "source_run": str(run.relative_to(root)),
        "paired_real_episode_pairs": 1, "completed_real_episodes": 2,
        "seed": cfg["seed"], "queries": len(rows), "independent_sample_count": None,
        "actual_episode_returns": actual_returns, "scientific_result": summary["scientific_result"],
        "rates": rates, "action_difference": {"signed_mean": float(action_delta.mean()),
            "mean_absolute": float(np.abs(action_delta).mean()), "rms": float(np.sqrt((action_delta ** 2).mean())),
            "maximum_absolute": float(np.abs(action_delta).max())},
        "phase_objective_gain": {"mean": float(gains.mean()), "min": float(gains.min()), "max": float(gains.max())},
        "tie_diagnostics": {"both_comparisons_have_unique_top_score_queries": len(unique_both),
            "unique_top_score_flip_count": sum(r["native_topk35_first_index_changed"] for r in unique_both),
            "native_top1_definition": "First index of native torch.topk(35), matching native CEM elite ordering; ties are unchanged native implementation behavior, not unique preference changes.",
            "at_least_one_tied_top_score_queries": len(rows) - len(unique_both)},
        "budget": {"real_environment_steps": summary["real_environment_steps"],
            "sampler_queries": sum(r["native_queries"] for r in summary["rows"]),
            "native_modelenv_forecast_batch_calls": k * sum(r["native_queries"] for r in summary["rows"]),
            "predictive_transitions": sum(r["predictive_transitions"] for r in summary["rows"]),
            "new_model_calls_for_this_analysis": 0, "new_environment_steps_for_this_analysis": 0},
        "interpretation": "These counterfactuals hold each actually visited intervention state, candidate bank and pre-final mean fixed, replacing only its final-row selection with the saved independent vector. They are not a second closed-loop trajectory, do not supply 200 independent samples, and do not establish harm when actual paired returns are equal.",
        "marginal_scope": "Exact scalar N-particle return-estimator bank-column permutations; not an empirical test of unrecorded individual forecast trajectories.",
        "source_paths": [str((run / "summary.json").relative_to(root)), str((arm / "episode.json").relative_to(root)),
                         str((run / "independent/episode.json").relative_to(root))]}
    with table.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
