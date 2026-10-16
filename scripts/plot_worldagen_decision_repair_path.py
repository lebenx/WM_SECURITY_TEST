#!/usr/bin/env python3
"""Plot deterministic Clean-anchor / Bus-Cut decision-repair path."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
stats = Path(os.environ.get(
    "DECISION_REPAIR_STATS_PATH",
    str(ROOT / "runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/decision_repair_path.json"),
))
stem = os.environ.get("DECISION_REPAIR_FIGURE_STEM", "worldagen_decision_repair_path")
x = json.loads(stats.read_text())
w = np.asarray(x["weights"], dtype=float)


def metric(key):
    return np.asarray([
        x["by_weight"][str(v)][key]["mean"]
        if isinstance(x["by_weight"][str(v)][key], dict)
        else x["by_weight"][str(v)][key]
        for v in w
    ])


reg = metric("regret")
recover = metric("clean_action_recovery_rate")
flip = metric("ranking_flip_vs_clean")
eef = metric("eef_delta_vs_clean")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
ax[0].plot(w, reg, marker="o", color="#b23a48")
ax[0].set(xlabel="Clean-anchor weight", ylabel="Regret (m)", title="Real consequence")
ax[1].plot(w, recover, marker="o", color="#2166ac", label="Clean action recovery")
ax[1].plot(w, flip, marker="s", color="#762a83", label="Pairwise flip vs Clean")
ax[1].set(xlabel="Clean-anchor weight", ylabel="Rate", title="Decision recovery")
ax[1].set_ylim(-.05, 1.05)
ax[1].legend(fontsize=7)
ax[2].plot(w, eef, marker="o", color="#e08214")
ax[2].axhline(0, color="black", lw=.8)
ax[2].set(xlabel="Clean-anchor weight", ylabel="EEF delta vs Clean (m)", title="Selected utility")
for a in ax:
    a.grid(axis="y", alpha=.25)
fig.suptitle("Deterministic Clean-anchor / Bus-Cut repair path (18 task-seed pairs, 162 Goal swaps)", fontsize=10)
fig.tight_layout()
for ext in ("svg", "pdf"):
    out = ROOT / "reports" / "figures" / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(out)
