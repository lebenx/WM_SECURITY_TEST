#!/usr/bin/env python3
"""Summarize WorldAgen goal-bus effects by recorded legal task text."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "worldagen_goal_bus_replicate_v2" / "summary.json"
OUT = ROOT / "runs" / "worldagen_goal_bus_task_groups_v0" / "summary.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    groups = defaultdict(list)
    for row in source["records"]:
        groups[row["recorded_goal"]].append(row)
    rows = []
    for task, records in sorted(groups.items()):
        native = [r["native"]["other_world"]["relative_l2"] for r in records]
        cut = [r["cut_repair"]["other_world"]["relative_l2"] for r in records]
        policy = [r["cut_repair"]["other_policy"]["relative_l2"] for r in records]
        rows.append({
            "task": task,
            "windows": len(records),
            "episodes": sorted({int(r["episode"]) for r in records}),
            "native_world_relative_l2_mean": sum(native) / len(native),
            "native_world_nonzero_all": all(x > 1e-12 for x in native),
            "cut_world_relative_l2_max": max(cut),
            "cut_world_exact_zero_all": all(x == 0.0 for x in cut),
            "cut_policy_relative_l2_mean": sum(policy) / len(policy),
        })
    result = {
        "status": "completed_worldagen_goal_bus_task_group_summary",
        "source": str(SOURCE.relative_to(ROOT)),
        "unique_recorded_tasks": len(rows),
        "windows": source["windows"],
        "native_nonzero_all_groups": all(r["native_world_nonzero_all"] for r in rows),
        "cut_exact_zero_all_groups": all(r["cut_world_exact_zero_all"] for r in rows),
        "scope": "Descriptive semantic grouping of the existing 30-window WorldAgen probe; repeated frames remain clustered within physical episodes.",
        "groups": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("unique_recorded_tasks", "windows", "native_nonzero_all_groups", "cut_exact_zero_all_groups")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
