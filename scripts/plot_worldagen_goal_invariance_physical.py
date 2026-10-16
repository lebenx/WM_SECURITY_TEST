#!/usr/bin/env python3
"""Plot the matched physical-target Goal-invariance audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--svg", required=True, type=Path)
    ap.add_argument("--pdf", required=True, type=Path)
    args = ap.parse_args()
    d = json.loads(args.input.read_text())
    offsets = [int(o) for o in d["offsets"]]
    by = d["by_offset"]
    means = np.asarray([by[str(o)]["swap_minus_clean_physical_mse"]["mean"] for o in offsets])
    cis = np.asarray([by[str(o)]["swap_minus_clean_physical_mse_task_cluster_bootstrap"]["ci95"] for o in offsets])
    err = np.vstack([means-cis[:, 0], cis[:, 1]-means])
    x = np.asarray([r["forecast_divergence_mean"] for r in d["records"]])
    y = np.asarray([r["swap_minus_clean_physical_mse"] for r in d["records"]])
    rho = d["aggregate"]["spearman_divergence_vs_physical_error_delta"]["rho"]
    p = d["aggregate"]["spearman_divergence_vs_physical_error_delta"]["pvalue"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(offsets, means, yerr=err, fmt="o", color="#245b8a", capsize=3)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Goal offset")
    ax.set_ylabel("Swap − Clean physical-target MSE")
    ax.set_title("Mixed-sign target error")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.scatter(x, y, s=13, alpha=0.55, color="#a6493c", edgecolors="none")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Swap−Clean latent forecast divergence")
    ax.set_ylabel("Swap−Clean physical-target MSE")
    ax.set_title(f"Rank correlation ρ={rho:.3f}, p={p:.3f}")
    ax.grid(alpha=0.25)
    fig.suptitle("Goal-Bus physical-target invariance audit", fontsize=11)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    fig.savefig(args.pdf, format="pdf")
    plt.close(fig)
    print(json.dumps({"svg": str(args.svg), "pdf": str(args.pdf), "n": len(x)}))


if __name__ == "__main__":
    main()
