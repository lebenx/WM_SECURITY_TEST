#!/usr/bin/env python3
"""Plot AlphaBrain future-output Goal effects across legal offsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--svg", type=Path, required=True)
    ap.add_argument("--pdf", type=Path, required=True)
    args = ap.parse_args()
    d = json.loads(args.input.read_text())
    offsets = [int(x) for x in d["offsets"]]
    names = ["future_proprio", "future_wrist", "future_primary"]
    labels = ["proprio", "wrist", "primary"]
    colors = ["#b85c38", "#4c78a8", "#59a14f"]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    for name, label, color in zip(names, labels, colors):
        vals = [d["per_offset"][str(o)]["native_goal_effect_abs_delta"][name]["mean"] for o in offsets]
        lo = [d["per_offset"][str(o)]["native_goal_effect_abs_delta"][name]["episode_cluster_bootstrap_ci95"][0] for o in offsets]
        hi = [d["per_offset"][str(o)]["native_goal_effect_abs_delta"][name]["episode_cluster_bootstrap_ci95"][1] for o in offsets]
        ax[0].errorbar(offsets, vals, yerr=[np.asarray(vals)-lo, np.asarray(hi)-vals], marker="o", capsize=3, label=label, color=color)
        vals = [d["per_offset"][str(o)]["native_goal_target_mse_delta"][name]["mean"] for o in offsets]
        lo = [d["per_offset"][str(o)]["native_goal_target_mse_delta"][name]["episode_cluster_bootstrap_ci95"][0] for o in offsets]
        hi = [d["per_offset"][str(o)]["native_goal_target_mse_delta"][name]["episode_cluster_bootstrap_ci95"][1] for o in offsets]
        ax[1].errorbar(offsets, vals, yerr=[np.asarray(vals)-lo, np.asarray(hi)-vals], marker="o", capsize=3, label=label, color=color)
    ax[0].set_title("Goal effect")
    ax[0].set_ylabel("absolute latent delta")
    ax[1].set_title("Target-error change")
    ax[1].set_ylabel("Goal-Swap minus Clean MSE")
    for a in ax:
        a.set_xlabel("legal Goal offset")
        a.set_xticks(offsets)
        a.axhline(0, color="0.4", lw=0.8)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("AlphaBrain fixed-state/fixed-action Goal audit (63 windows)")
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    fig.savefig(args.pdf, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
