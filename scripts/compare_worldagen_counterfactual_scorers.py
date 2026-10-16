#!/usr/bin/env python3
"""Compare fixed WorldAgen counterfactual ranking runs from two scorers.

The comparison is deliberately descriptive: rows are aligned by task_id and
condition, and no claim of significance is made.  It is useful for checking
whether a decision-level effect survives a change in the independent planner
score reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if data.get("selection_target_fixed_to_original_goal") is not True:
        raise ValueError(f"{path}: score target is not fixed to the original goal")
    if data.get("candidate_set_frozen_across_conditions") is not True:
        raise ValueError(f"{path}: candidate set is not frozen")
    return data


def _rows(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in data["rows"]}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _condition_metrics(rows: dict[int, dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = []
    eef = []
    eef_delta = []
    regrets = []
    for task_id, row in rows.items():
        selected.append(row["selected"][condition])
        replay = row["replays"][condition]
        eef.append(replay["eef_approach_progress"])
        eef_delta.append(replay["eef_approach_progress"] - row["replays"]["clean"]["eef_approach_progress"])
        if replay.get("eef_approach_regret_vs_candidate_best") is not None:
            regrets.append(replay["eef_approach_regret_vs_candidate_best"])
    return {
        "selected": {str(k): v for k, v in sorted(zip(rows, selected))},
        "eef_approach_progress_mean": _mean(eef),
        "eef_approach_delta_vs_clean_mean": _mean(eef_delta),
        "regret_vs_candidate_best_mean": _mean(regrets),
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ar, br = _rows(a), _rows(b)
    if set(ar) != set(br):
        raise ValueError(f"task mismatch: {sorted(ar)} vs {sorted(br)}")
    conditions = sorted(set(ar[next(iter(ar))]["selected"]) & set(br[next(iter(br))]["selected"]))
    out: dict[str, Any] = {
        "scorers": {
            "a": a.get("clip_reconstruction", "unspecified"),
            "b": b.get("clip_reconstruction", "unspecified"),
        },
        "task_ids": sorted(ar),
        "conditions": conditions,
        "metrics": {},
        "selected_action_disagreement_count": {},
    }
    for condition in conditions:
        out["metrics"][condition] = {
            "a": _condition_metrics(ar, condition),
            "b": _condition_metrics(br, condition),
        }
        out["selected_action_disagreement_count"][condition] = sum(
            ar[t]["selected"][condition] != br[t]["selected"][condition] for t in ar
        )

    control_conditions = [c for c in conditions if c.endswith("control_cut")]
    bus_conditions = [c for c in conditions if c.endswith("bus_cut")]
    out["control_cut_score_identity"] = {
        "a": a.get("statistics", {}).get("control_cut_score_identity"),
        "b": b.get("statistics", {}).get("control_cut_score_identity"),
    }
    out["decision_recovery"] = {
        "a": a.get("statistics", {}).get("decision_recovery"),
        "b": b.get("statistics", {}).get("decision_recovery"),
    }
    out["interpretation"] = (
        "Descriptive scorer-sensitivity comparison. The two runs share task ids, "
        "candidate trajectories, fixed original-goal text, replay protocol, and seed; "
        "utility direction is not treated as scorer-invariant."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(_load(args.run_a), _load(args.run_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
