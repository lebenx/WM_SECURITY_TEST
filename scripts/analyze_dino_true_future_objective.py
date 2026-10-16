#!/usr/bin/env python3
"""Score cached actual futures with the unchanged DINO planning objective.

No model, encoder, environment, checkpoint, or API is called. The output is a
post-result proxy-alignment diagnostic, not a new attack trial.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/dino_action_saturation_gate_20261015"
NEW = ROOT / "runs/dino_action_saturation_exploratory_20261015"


def main(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    import numpy as np
    import torch
    from tensordict import TensorDict

    sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
    from pilot_dino_metaworld_clean_gate import cfg_for
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    prior = json.loads((OLD / "summary.json").read_text())
    result = json.loads((NEW / "summary.json").read_text())
    assert result["status"] == "completed_exploratory_cached_alias_forecasts"
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    with np.load(OLD / "forecasts_and_encodings.npz", allow_pickle=False) as z:
        truth = TensorDict({k: torch.from_numpy(z["ground_truth_" + k].copy()) for k in ("visual", "proprio")})
        goal = TensorDict({k: torch.from_numpy(z["goal_" + k].copy()) for k in ("visual", "proprio")})
    with np.load(OLD / "frozen_bank.npz", allow_pickle=False) as z:
        actions = torch.from_numpy(z["normalized"][0].copy()).reshape(5, 6, 20).permute(1, 0, 2).contiguous()
    assert all(tuple(truth[k].shape[:2]) == (6, 5) for k in ("visual", "proprio"))
    assert all(tuple(goal[k].shape[:2]) == (1, 1) for k in ("visual", "proprio"))
    cfg = cfg_for(NEW, "cpu")
    objective = ReprTargetDistMPCObjective(cfg, target_enc=goal, **cfg.planner.planning_objective)
    assert objective.alpha == .1 and objective.sum_all_diffs is False
    with patch.object(torch, "load", side_effect=RuntimeError("No checkpoint loading allowed")), torch.inference_mode():
        values = objective(truth, actions)
    assert tuple(values.shape) == (5,) and torch.isfinite(values).all()
    cost = values.tolist()
    ranking = sorted(range(5), key=lambda i: (cost[i], i))
    selected = ranking[0]
    utilities = prior["utilities"]
    real_order = sorted(range(5), key=lambda i: (-utilities[i]["primary30"]["terminal_native_reward"], i))
    comparisons = {}
    for arm, r in result["conditions"].items():
        chosen, c, order = r["selected_candidate"], r["native_costs"], r["ranking"]
        comparisons[arm] = {
            "native_costs": c, "ranking": order, "selected_candidate": chosen,
            "best_vs_runner_up_cost_margin": c[order[1]] - c[order[0]],
            "forecast_cost_minus_true_future_cost_per_candidate": [x - y for x, y in zip(c, cost)],
            "pairwise_order_changes_vs_true_future": sum(
                (order.index(i) < order.index(j)) != (ranking.index(i) < ranking.index(j))
                for i in range(5) for j in range(i + 1, 5)),
            "selected_true_objective_cost_regret": cost[chosen] - cost[selected],
            "selected_real_utility": utilities[chosen],
            "selected_real_terminal_reward_minus_true_objective_selected": (
                utilities[chosen]["primary30"]["terminal_native_reward"]
                - utilities[selected]["primary30"]["terminal_native_reward"]),
        }
    report = {
        "schema_version": 1,
        "status": "completed_cached_true_future_native_objective_comparator",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-result diagnostic on the same one development root and five frozen candidates. No new forecasts, candidate changes, or endpoint changes.",
        "sources": {
            "prior_summary": str((OLD / "summary.json").relative_to(ROOT)),
            "cached_truth_goal": str((OLD / "forecasts_and_encodings.npz").relative_to(ROOT)),
            "frozen_actions": str((OLD / "frozen_bank.npz").relative_to(ROOT)),
            "forecast_summary": str((NEW / "summary.json").relative_to(ROOT)),
            "native_objective": "vendor/jepa-wms/evals/simu_env_planning/planning/planning/objectives.py",
            "native_config_loader": "scripts/pilot_dino_metaworld_clean_gate.py",
        },
        "method": {
            "class": "ReprTargetDistMPCObjective", "sum_all_diffs": False, "alpha": float(objective.alpha),
            "truth_time_alignment": "Six actual-future encoded observations at5/10/15/20/25/30 physical controls; score only the final30-control endpoint. No initial prefix is required by this terminal-only objective.",
            "goal": "Same cached native target encoding as all three forecasts",
            "actions_shape": list(actions.shape),
            "truth_shapes": {k: list(truth[k].shape) for k in ("visual", "proprio")},
            "goal_shapes": {k: list(goal[k].shape) for k in ("visual", "proprio")},
            "dtype": "float32", "device": "cpu",
        },
        "true_future_native_objective": {
            "costs": cost, "ranking": ranking, "selected_candidate": selected,
            "best_vs_runner_up_cost_margin": cost[ranking[1]] - cost[ranking[0]],
            "exact_minimum_ties": sum(v == min(cost) for v in cost),
            "selected_real_utility": utilities[selected],
        },
        "real_terminal_reward_reference": {
            "ranking": real_order, "best_candidate": real_order[0],
            "best_terminal_reward": utilities[real_order[0]]["primary30"]["terminal_native_reward"],
            "true_objective_selected_regret": (
                utilities[real_order[0]]["primary30"]["terminal_native_reward"]
                - utilities[selected]["primary30"]["terminal_native_reward"]),
        },
        "forecast_comparisons": comparisons,
        "activity": {"native_objective_calls": 1, "model_calls": 0, "unroll_calls": 0, "encoder_calls": 0,
                     "environment_steps": 0, "checkpoint_loads": 0, "api_calls": 0,
                     "cuda_initialized": torch.cuda.is_initialized()},
        "limitations": [
            "This evaluates the native latent-distance proxy on actual outcomes; it is not a newly trained/calibrated true-value model.",
            "Cost margins are deterministic finite-bank differences, not calibrated uncertainty or statistical significance.",
            "The diagnostic cannot assign a general fraction of decision harm to objective misalignment versus forecast error.",
        ],
    }
    with out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(out), "costs": cost, "ranking": ranking, "selected_candidate": selected}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NEW / "oracle_objective_analysis.json")
    main(parser.parse_args().out.resolve())
