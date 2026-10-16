#!/usr/bin/env python3
"""Render the legal-goal-offset robustness figure used by the report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json"
OUT = ROOT / "reports/figures/worldagen_goal_bus_offset_sweep.svg"


def main() -> None:
    data = json.loads(STATS.read_text())
    offsets = [int(x) for x in data["offsets"]]
    by = data["by_offset"]
    world = [by[str(x)]["native_world_relative_l2_mean"] for x in offsets]
    policy = [by[str(x)]["cut_policy_relative_l2_mean"] for x in offsets]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(offsets, world, marker="o", linewidth=2.2, label="native world Δ (relative L2)")
    ax.plot(offsets, policy, marker="s", linewidth=2.0, label="edge-cut policy Δ (relative L2)")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Legal task offset")
    ax.set_ylabel("Mean paired relative L2")
    ax.set_xticks(offsets)
    ax.set_title("WorldAgen shared action-token bus across all legal goal offsets")
    ax.legend(frameon=True, fontsize=9)
    ax.text(
        0.02, 0.03,
        "edge-cut world Δ = 0 for every offset;\n95% clustered CI (world): "
        f"[{data['native_world_relative_l2_cluster_bootstrap_95ci'][0]:.3f}, "
        f"{data['native_world_relative_l2_cluster_bootstrap_95ci'][1]:.3f}]",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
