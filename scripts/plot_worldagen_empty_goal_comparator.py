#!/usr/bin/env python3
"""Plot the 50-state empty-Goal comparator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--svg", required=True, type=Path)
    ap.add_argument("--pdf", required=True, type=Path)
    args = ap.parse_args()
    d = json.loads(args.input.read_text())
    rows = d["rows"]
    state_values = np.asarray(d["empty_minus_clean_target_mse_by_state"], dtype=float)
    action_values = np.asarray([r["empty_minus_clean_target_mse"] for r in rows], dtype=float)
    ci = np.asarray(d["empty_minus_clean_target_mse_state_cluster_bootstrap_ci95"], dtype=float)
    mean = float(d["empty_minus_clean_target_mse_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), constrained_layout=True)
    ax = axes[0]
    ax.hist(state_values, bins=10, color="#245b8a", alpha=0.85, edgecolor="white")
    ax.axvline(mean, color="#a6493c", linewidth=2, label=f"mean={mean:.4f}")
    ax.axvspan(ci[0], ci[1], color="#a6493c", alpha=0.15, label=f"cluster CI [{ci[0]:.4f},{ci[1]:.4f}]")
    ax.set_xlabel("Empty Goal − Clean target MSE per initial state")
    ax.set_ylabel("Number of initial states")
    ax.set_title("50-state comparator")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.boxplot(action_values, vert=True, patch_artist=True,
               boxprops={"facecolor": "#8bb7d9"}, medianprops={"color": "#a6493c", "linewidth": 2})
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks([1], ["4 fixed actions"])
    ax.set_ylabel("Target-MSE delta")
    ax.set_title("All 200 target replays")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("WorldAgen same-scene empty-Goal diagnostic", fontsize=11)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    fig.savefig(args.pdf, format="pdf")
    plt.close(fig)
    print(json.dumps({"svg": str(args.svg), "pdf": str(args.pdf), "states": len(state_values), "replays": int(action_values.size)}))


if __name__ == "__main__":
    main()
