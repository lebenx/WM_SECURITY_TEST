#!/usr/bin/env python3
"""Plot the fixed-planner decision and repair evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "worldagen_counterfactual_action_ranking_v0"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

base = json.loads((RUN / "summary_fixedscore_v2_tasks6_statistics.json").read_text())["statistics"]
patch = json.loads((RUN / "summary_activation_patch_tasks6_v2_statistics.json").read_text())["statistics"]

labels = ["Goal-Swap", "Bus-Cut", "ActPatch L4", "ActPatch L8"]
conditions = [
    "goal_swap_offset_1",
    "goal_swap_offset_1_bus_cut",
    "goal_swap_activation_patch_l4",
    "goal_swap_activation_patch_l8",
]
colors = ["#b23a48", "#2166ac", "#762a83", "#1b7837"]

eef = [
    base["eef_approach_progress_delta_vs_clean"][conditions[0]]["mean"],
    base["eef_approach_progress_delta_vs_clean"][conditions[1]]["mean"],
    patch["eef_approach_progress_delta_vs_clean"][conditions[2]]["mean"],
    patch["eef_approach_progress_delta_vs_clean"][conditions[3]]["mean"],
]
eef_sd = [
    base["eef_approach_progress_delta_vs_clean"][conditions[0]]["sd"],
    base["eef_approach_progress_delta_vs_clean"][conditions[1]]["sd"],
    patch["eef_approach_progress_delta_vs_clean"][conditions[2]]["sd"],
    patch["eef_approach_progress_delta_vs_clean"][conditions[3]]["sd"],
]
flip = [
    base["ranking_flip_rate"][conditions[0]]["mean"],
    base["ranking_flip_rate"][conditions[1]]["mean"],
    patch["ranking_flip_rate"][conditions[2]]["mean"],
    patch["ranking_flip_rate"][conditions[3]]["mean"],
]
residual = [
    patch["forecast_relative_l2_to_clean"]["goal_swap"]["mean"],
    patch["forecast_relative_l2_to_clean"]["goal_swap_bus_cut"]["mean"],
    patch["forecast_relative_l2_to_clean"][conditions[2]]["mean"],
    patch["forecast_relative_l2_to_clean"][conditions[3]]["mean"],
]
recovery = [
    0.0,
    patch["decision_recovery"]["goal_swap_bus_cut"]["matches_clean_rate"],
    patch["decision_recovery"][conditions[2]]["matches_clean_rate"],
    patch["decision_recovery"][conditions[3]]["matches_clean_rate"],
]

fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8))
for ax, vals, title, ylabel in [
    (axes[0], eef, "Real EEF consequence", "EEF progress delta vs Clean (m)"),
    (axes[1], residual, "Forecast residual", "World forecast relative L2 to Clean"),
    (axes[2], flip, "Ranking perturbation", "Pairwise ranking flip rate"),
    (axes[3], recovery, "Clean action recovery", "Fraction of tasks"),
]:
    x = np.arange(len(labels))
    if ax is axes[0]:
        ax.bar(x, vals, yerr=eef_sd, color=colors, alpha=0.9, capsize=3)
        ax.axhline(0, color="black", lw=0.8)
    else:
        ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "Goal-Bus decision test: forecast repair reduces residual but does not ensure clean selection",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(OUT / "goal_bus_decision_evidence.svg", bbox_inches="tight")
fig.savefig(OUT / "goal_bus_decision_evidence.pdf", bbox_inches="tight")
print(OUT / "goal_bus_decision_evidence.svg")
print(OUT / "goal_bus_decision_evidence.pdf")
