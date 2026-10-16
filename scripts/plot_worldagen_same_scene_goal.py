#!/usr/bin/env python3
"""Plot the same-scene Goal crossing target-error control."""
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
    state_values = np.asarray(d["native_nonclean_target_mse_delta_vs_clean_by_state"], dtype=float)
    goal_values = np.asarray(d["native_nonclean_target_mse_delta_by_goal_mean"], dtype=float)
    clean_idx = int(d["clean_goal_index"])
    goals = [i for i in range(len(goal_values) + 1) if i != clean_idx]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(state_values))
    ax.bar(x, state_values, color="#245b8a", width=0.65)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, [f"state {i}" for i in range(len(state_values))])
    ax.set_ylabel("Non-Clean Goal − Clean target MSE")
    ax.set_title("All shared states positive")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    x = np.arange(len(goal_values))
    colors = ["#a6493c" if v >= 0 else "#777777" for v in goal_values]
    ax.bar(x, goal_values, color=colors, width=0.65)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, [f"G{i}" for i in goals])
    ax.set_ylabel("Target-MSE delta")
    ax.set_title("Goal means (clean G3 omitted)")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Same-scene Goal crossing: physical target control", fontsize=11)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    fig.savefig(args.pdf, format="pdf")
    plt.close(fig)
    print(json.dumps({"svg": str(args.svg), "pdf": str(args.pdf), "states": len(state_values), "goals": len(goal_values)}))


if __name__ == "__main__":
    main()
