#!/usr/bin/env python3
"""Plot WorldAgen action-preservation and task-group evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
action = json.loads((ROOT / "runs/worldagen_goal_bus_action_control_v1/summary.json").read_text())
groups = json.loads((ROOT / "runs/worldagen_goal_bus_task_groups_v0/summary.json").read_text())
outdir = ROOT / "reports/figures"
outdir.mkdir(parents=True, exist_ok=True)

rows = action["rows"]
x = np.arange(len(rows))
native = np.array([r["native_action_world"]["relative_l2"] for r in rows])
cut = np.array([r["cut_action_world"]["relative_l2"] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.25), gridspec_kw={"width_ratios": [1.35, 1.0]})
ax = axes[0]
ax.plot(x, native, "o-", lw=1.4, ms=3.5, color="#b23a48", label="native action→world")
ax.plot(x, cut, "s-", lw=1.4, ms=3.2, color="#2166ac", label="cut action→world")
ax.set_xlabel("Fixed window (episode ordered; frame 30/60/90)")
ax.set_ylabel("Relative L2")
ax.set_title("WorldAgen: real action control\n30 fixed image/state/goal windows")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
labels = []
vals = []
for row in groups["groups"]:
    text = row["task"]
    short = text.replace("put ", "").replace(" and put ", " + ")
    if len(short) > 34:
        short = short[:31] + "…"
    labels.append(short)
    vals.append(row["native_world_relative_l2_mean"])
y = np.arange(len(labels))
ax.barh(y, vals, color="#b23a48", alpha=0.9, label="native goal swap")
ax.scatter(np.zeros_like(y), y, marker="s", s=22, color="#2166ac", label="goal-bus cut (exact zero)", zorder=3)
ax.set_yticks(y, labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Mean relative L2")
ax.set_title("Six legal task groups")
ax.grid(axis="x", alpha=0.25)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

fig.suptitle("Repair removes instruction dependence while retaining physical action sensitivity", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
for suffix in ("svg", "pdf"):
    fig.savefig(outdir / f"worldagen_action_control.{suffix}", bbox_inches="tight")
print(outdir / "worldagen_action_control.svg")
print(outdir / "worldagen_action_control.pdf")
