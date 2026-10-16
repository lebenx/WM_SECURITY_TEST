#!/usr/bin/env python3
"""Plot measured finite-cache diagnostics; no model or simulator is invoked."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/decision_leverage_paper"
SOURCES = ["reports/cosmos_first_choice_identifiability_bounds_20261015.json",
           "reports/dino_first_query_terminal_truth_decision_20261015.json"]


def main():
    cosmos, dino = [json.loads((ROOT / p).read_text()) for p in SOURCES]
    interval = cosmos["any_first_choice_selector"]["success_count_or_expectation_interval"]
    native = cosmos["relative_to_recorded_native"]["native_success_count"]
    rows = dino["rows"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), layout="constrained")
    colors = ["#9da7b1", "#286a9b", "#6a9d70"]
    values = [interval[0], native, interval[1]]
    bars = axes[0].bar(["Worst branch", "Native", "Best branch"], values, color=colors)
    axes[0].bar_label(bars, padding=3, fmt="%.0f")
    axes[0].set_ylim(0, 33)
    axes[0].set_ylabel("Successful recorded continuations / 30")
    axes[0].set_title("A. Cosmos: fixed first-choice support")
    axes[0].tick_params(axis="x", labelsize=9)
    x = np.arange(2)
    a = [r["original_forecast_weighted_mse"] for r in rows]
    b = [r["canonical_forecast_weighted_mse"] for r in rows]
    bars_a = axes[1].bar(x-.18, a, .36, label="Original request", color="#286a9b")
    bars_b = axes[1].bar(x+.18, b, .36, label="Canonical request", color="#c87f3e")
    axes[1].bar_label(bars_a, padding=3, fmt="%.3f", fontsize=9)
    axes[1].bar_label(bars_b, padding=3, fmt="%.3f", fontsize=9)
    axes[1].set_xticks(x, ["Action a0", "Action a1"])
    axes[1].set_ylim(0, 1.9)
    axes[1].set_ylabel("Forecast-to-real-endpoint feature MSE")
    axes[1].set_title("B. DINO: accuracy improves for both")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    utilities = [r["cumulative_native_reward"] for r in rows]
    bars = axes[2].bar(["Original\nforecasts", "Canonical\nforecasts", "Real future\nencodings"],
                       [utilities[0], utilities[1], utilities[1]],
                       color=["#286a9b", "#c87f3e", "#6a9d70"])
    axes[2].bar_label(bars, padding=3, fmt="%.2f")
    axes[2].set_ylim(0, 112)
    axes[2].set_ylabel("Selected action's real 30-control return")
    axes[2].set_title("C. DINO: fixed proxy selects lower utility")
    axes[2].tick_params(axis="x", labelsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"decision_diagnostics.{suffix}", dpi=180)
    plt.close(fig)
    provenance = {
        "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in SOURCES],
        "scope": "Panel A: 30 development roots with fixed candidate continuations. Panels B/C: the same single post-hoc DINO root and two mean-action sequences; no CI or independent replication implied.",
        "cosmos_success_values": values,
        "dino_original_mse": a, "dino_canonical_mse": b,
        "dino_selected_return": [utilities[0], utilities[1], utilities[1]],
        "new_model_calls": 0, "new_environment_steps": 0,
    }
    (OUT / "decision_diagnostics.sources.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(OUT / "decision_diagnostics.pdf")


if __name__ == "__main__":
    main()
