#!/usr/bin/env python3
"""Plot AlphaBrain Goal effects on future proprio/wrist/primary slots."""
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
    slots = ["future_proprio", "future_wrist", "future_primary"]
    labels = ["Future proprio", "Future wrist", "Future primary"]
    means = [d["native_goal_effect_abs_delta"][s]["mean"] for s in slots]
    lo = [d["native_goal_effect_abs_delta"][s]["episode_cluster_bootstrap_ci95"][0] for s in slots]
    hi = [d["native_goal_effect_abs_delta"][s]["episode_cluster_bootstrap_ci95"][1] for s in slots]
    err = np.asarray([means, np.asarray(means) - lo, np.asarray(hi) - means])
    mse = [d["native_goal_target_mse_delta"][s]["mean"] for s in slots]
    mse_lo = [d["native_goal_target_mse_delta"][s]["episode_cluster_bootstrap_ci95"][0] for s in slots]
    mse_hi = [d["native_goal_target_mse_delta"][s]["episode_cluster_bootstrap_ci95"][1] for s in slots]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    x = np.arange(len(slots))
    ax[0].bar(x, means, color="#b85c38")
    ax[0].errorbar(x, means, yerr=err[1:], fmt="none", ecolor="black", capsize=3, lw=1)
    ax[0].set_title("Goal effect on forecast")
    ax[0].set_ylabel("mean absolute latent delta")
    ax[0].set_xticks(x, labels, rotation=18, ha="right")
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[1].bar(x, mse, color="#4c78a8")
    ax[1].errorbar(x, mse, yerr=[np.asarray(mse)-mse_lo, np.asarray(mse_hi)-mse], fmt="none", ecolor="black", capsize=3, lw=1)
    ax[1].set_title("Change in target MSE")
    ax[1].set_ylabel("Goal-Swap minus Clean")
    ax[1].set_xticks(x, labels, rotation=18, ha="right")
    ax[1].axhline(0, color="0.4", lw=0.8)
    fig.suptitle("AlphaBrain: fixed-state/fixed-action Goal intervention (21 windows)")
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    fig.savefig(args.pdf, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
