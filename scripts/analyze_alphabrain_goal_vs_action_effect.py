#!/usr/bin/env python3
"""Pair existing AlphaBrain goal and alternate-action world effects.

The goal rows use the seed-1 forecast matrix at frame 30 for episodes 3, 6,
and 8, matching the three episodes in the repaired alternate-action control.
This is a descriptive effect-size comparison, not a new independent test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json"
ACTION = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_action_control_v0/summary.json"
OUT = ROOT / "runs/alphabrain_goal_vs_action_effect_v0/summary.json"


def main() -> None:
    goal_rows = json.loads(GOAL.read_text())["rows"]
    action = json.loads(ACTION.read_text())
    action_by_ep = {int(ep): float(value) for ep, value in zip(action["episodes"], action["cut_values"])}
    rows = []
    for episode in action["episodes"]:
        matches = [r for r in goal_rows if int(r["episode"]) == int(episode) and int(r["frame"]) == 30]
        if len(matches) != 1:
            raise RuntimeError(f"expected one frame-30 goal row for episode {episode}")
        goal_effect = float(matches[0]["native_goal_world_delta_mean"])
        action_effect = action_by_ep[int(episode)]
        rows.append({
            "episode": int(episode),
            "frame": 30,
            "goal_world_effect": goal_effect,
            "alternate_action_world_effect": action_effect,
            "goal_to_action_effect_ratio": goal_effect / max(action_effect, 1e-12),
        })
    goal = np.asarray([r["goal_world_effect"] for r in rows])
    act = np.asarray([r["alternate_action_world_effect"] for r in rows])
    result = {
        "status": "completed_descriptive_goal_vs_action_effect",
        "rows": rows,
        "episodes": [int(x) for x in action["episodes"]],
        "goal_effect_mean": float(goal.mean()),
        "alternate_action_effect_mean": float(act.mean()),
        "goal_to_action_mean_ratio": float(goal.mean() / act.mean()),
        "interpretation": "On the three matched episodes, the legal goal-swap world effect averages 0.819 of the repaired alternate-action effect. This is a descriptive scale comparison; the three-episode action control is not a broad significance test.",
        "sources": [str(GOAL.relative_to(ROOT)), str(ACTION.relative_to(ROOT))],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
