#!/usr/bin/env python3
"""Run the frozen WorldAgen goal-bus gate for all legal task offsets."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFSETS = [2, 3, 4, 6, 7, 8, 9]
DATA_ROOTS = "/home/xhj/WM_Security/data/worldagen_goal_probe,/home/xhj/WM_Security/data/worldagen_goal_probe_extra"
OUT_ROOT = ROOT / "runs/worldagen_goal_bus_offset_sweep_v1"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in OFFSETS:
        out = OUT_ROOT / f"offset_{offset}.json"
        log = OUT_ROOT / f"offset_{offset}.log"
        env = os.environ.copy()
        env.update({
            "LIBERO_CONFIG_PATH": "/tmp/libero_cfg",
            "WORLDAGEN_GOAL_OFFSET": str(offset),
            "WORLDAGEN_DATA_ROOTS": DATA_ROOTS,
            "WORLDAGEN_OUT": str(out),
            "PYTHONPATH": f"{ROOT / 'vendor/LIBERO'}:{ROOT / 'vendor/WorldAgen'}",
        })
        cmd = [str(ROOT / ".venv-worldagen/bin/python"), str(ROOT / "scripts/pilot_worldagen_goal_bus_replicate.py")]
        with log.open("w") as handle:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        data = json.loads(out.read_text())
        rows.append({"offset": offset, "source": str(out.relative_to(ROOT)), "aggregate": data["aggregate"]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = {
        "status": "completed_worldagen_goal_bus_offset_sweep",
        "offsets": OFFSETS,
        "windows_per_offset": 30,
        "sources": rows,
        "scope": "Frozen native forward over the expanded real LIBERO windows; no training, environment rollout, or task-harm claim.",
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
