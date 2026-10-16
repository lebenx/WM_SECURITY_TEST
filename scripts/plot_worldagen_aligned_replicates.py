#!/usr/bin/env python3
"""Plot the three-seed calibrated planner replication."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
stats_path = Path(os.environ.get(
    "ALIGNED_STATS_PATH",
    str(ROOT / "runs/worldagen_aligned_planner_replicates_v1/statistics.json"),
))
figure_stem = os.environ.get("ALIGNED_FIGURE_STEM", "worldagen_aligned_planner_replicates")
stats = json.loads(stats_path.read_text())
s = stats["per_scorer"]["reference_scalar"]
labels = ["Clean", "Swap 1", "Swap 5", "Swap 1 + Bus-Cut", "Swap 5 + Bus-Cut"]
conds = ["clean", "swap_1", "swap_5", "swap_1_bus_cut", "swap_5_bus_cut"]
reg = [s[c]["regret_vs_candidate_best"]["mean"] for c in conds]
reg_sd = [s[c]["regret_vs_candidate_best"]["sd"] for c in conds]
changed = [s[c]["selected_action_changed_vs_clean"]["mean"] for c in conds]
random_reg = stats["candidate_random_baseline"]["expected_regret"]["mean"]

effect_labels = ["Swap 1", "Swap 5", "Swap 1 + Bus-Cut", "Swap 5 + Bus-Cut"]
effect_conds = ["swap_1", "swap_5", "swap_1_bus_cut", "swap_5_bus_cut"]
effect = [s[c]["eef_delta_vs_clean_selected_action"]["mean"] for c in effect_conds]
cis = [s[c]["task_cluster_bootstrap_eef_delta_ci95"] ["ci95"] for c in effect_conds]
yerr = np.asarray([[m - lo for m, (lo, hi) in zip(effect, cis)], [hi - m for m, (lo, hi) in zip(effect, cis)]])

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
x = np.arange(len(labels))
ax[0].bar(x, reg, yerr=reg_sd, capsize=3, color=["#4d4d4d", "#b23a48", "#b23a48", "#2166ac", "#2166ac"])
ax[0].axhline(random_reg, color="black", ls="--", lw=1, label="random expected regret")
ax[0].set_title("Held-out regret")
ax[0].set_ylabel("EEF regret (m)")
ax[0].set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
ax[0].legend(fontsize=7)
ax[1].bar(x, changed, color=["#4d4d4d", "#b23a48", "#b23a48", "#2166ac", "#2166ac"])
ax[1].set_title("Selected action changed")
ax[1].set_ylabel("Fraction of 18 task-seed pairs")
ax[1].set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
ax[2].bar(np.arange(len(effect_labels)), effect, yerr=yerr, capsize=3, color=["#b23a48", "#b23a48", "#2166ac", "#2166ac"])
ax[2].axhline(0, color="black", lw=.8)
ax[2].set_title("EEF delta vs Clean selected action")
ax[2].set_ylabel("EEF progress delta (m)")
ax[2].set_xticks(np.arange(len(effect_labels)), effect_labels, rotation=38, ha="right", fontsize=8)
for a in ax:
    a.grid(axis="y", alpha=.25)
fig.suptitle("WorldAgen aligned planner: three seeds, six physical tasks", fontsize=11)
fig.tight_layout()
for ext in ("svg", "pdf"):
    out = ROOT / "reports" / "figures" / f"{figure_stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(out)
