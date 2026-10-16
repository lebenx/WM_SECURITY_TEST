#!/usr/bin/env python3
"""Aggregate the matching LIBERO-Goal clean-task pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "alphabrain_cosmos_libero_goal_control_v0"


def main() -> None:
    paths = sorted(RUN_DIR.glob("task*_304.json"))
    if not paths:
        raise SystemExit(f"no task*_304.json files under {RUN_DIR}")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    results = [item for run in runs for item in run["results"]]
    native = [r for r in results if r["condition"] == "native"]
    cut = [r for r in results if r["condition"] == "fixed_action_world_cut"]
    task_specs = []
    for path, run in zip(paths, runs):
        task_id = run.get("task_id")
        if task_id is None:
            task_id = int(path.stem.split("_")[0].removeprefix("task"))
        task_specs.append({"task_id": task_id, "task": run["task"]})
    summary = {
        "status": "completed_alphabrain_matching_goal_clean_task_aggregate",
        "suite": "libero_goal",
        "tasks": task_specs,
        "task_count": len(runs),
        "horizon_steps": 304,
        "native_successes": f"{sum(bool(r['success']) for r in native)}/{len(native)}",
        "fixed_action_world_cut_successes": f"{sum(bool(r['success']) for r in cut)}/{len(cut)}",
        "native_action_norm_mean": sum(r["action_norm_mean"] for r in native) / len(native),
        "fixed_action_world_cut_action_norm_mean": sum(r["action_norm_mean"] for r in cut) / len(cut),
        "source_runs": [str(p.relative_to(ROOT)) for p in paths],
        "interpretation": "Matching dataset_mix=libero_goal clean capability gate only; no task-harm attribution.",
    }
    out = RUN_DIR / "aggregate_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
