#!/usr/bin/env python3
"""Plot the legal-goal dose-response sweep."""
from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
stats_path = Path(os.environ.get(
    "GOAL_SWEEP_STATS_PATH",
    str(ROOT / "runs/worldagen_aligned_goal_sweep_h30_v1/statistics_sweep.json"),
))
stem = os.environ.get("GOAL_SWEEP_FIGURE_STEM", "worldagen_aligned_goal_sweep")
x = json.loads(stats_path.read_text())
offsets = np.asarray(x["offsets"], dtype=int)
task_count = len(x.get("task_ids", []))
pair_count = x.get("task_seed_pairs")
eef = np.asarray([x["by_offset"][str(o)]["eef_delta_vs_clean"]["mean"] for o in offsets])
eef_sd = np.asarray([x["by_offset"][str(o)]["eef_delta_vs_clean"]["sd"] for o in offsets])
reg = np.asarray([x["by_offset"][str(o)]["regret"]["mean"] for o in offsets])
bus_reg = np.asarray([x["by_offset"][str(o)]["bus_cut_regret"]["mean"] for o in offsets])
changed_key = "selected_changed_rate" if "selected_changed_rate" in x["by_offset"][str(offsets[0])] else "selected_changed"
changed = np.asarray([
    x["by_offset"][str(o)][changed_key]["mean"] if isinstance(x["by_offset"][str(o)][changed_key], dict)
    else x["by_offset"][str(o)][changed_key]
    for o in offsets
])
flip_key = "ranking_flip_rate"
flip = np.asarray([
    x["by_offset"][str(o)][flip_key]["mean"]
    if flip_key in x["by_offset"][str(o)] else np.nan
    for o in offsets
])

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].errorbar(offsets, eef, yerr=eef_sd, marker="o", color="#b23a48", capsize=3)
ax[0].axhline(0, color="black", lw=.8)
ax[0].set(title="Real EEF consequence", xlabel="Legal Goal offset", ylabel="EEF delta vs Clean (m)")
ax[1].plot(offsets, reg, marker="o", color="#b23a48", label="Goal-Swap")
ax[1].plot(offsets, bus_reg, marker="s", color="#2166ac", label="Bus-Cut")
ax[1].set(title="Candidate regret", xlabel="Legal Goal offset", ylabel="Regret (m)")
ax[1].legend(fontsize=8)
ax[2].plot(offsets, flip, marker="o", color="#762a83", label="pairwise rank flip")
ax[2].plot(offsets, changed, marker="s", color="#e08214", label="selected action changed")
ax[2].set(title="Ranking intervention", xlabel="Legal Goal offset", ylabel="Rate")
ax[2].set_ylim(-.05, 1.05)
ax[2].legend(fontsize=7)
for a in ax: a.grid(axis="y", alpha=.25)
subtitle = f"Goal-Bus legal-goal dose response: {task_count} LIBERO tasks, 30-step replay"
if pair_count:
    subtitle += f", {pair_count} task-seed pairs"
fig.suptitle(subtitle, fontsize=11)
fig.tight_layout()
for ext in ("svg", "pdf"):
    out = ROOT / "reports" / "figures" / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(out)
