#!/usr/bin/env python3
"""Plot the completed fixed-bank truth countercheck; no new experiment calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "dino_terminal_truth_countercheck_20261015"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    report = json.loads((ROOT / "reports/dino_first_query_terminal_truth_decision_20261015.json").read_text())
    assert report["status"] == "completed_full_horizon_truth_countercheck_no_prediction_degradation_claim"
    rows = report["rows"]
    assert len(rows) == 2 and all(row["controls"] == 30 for row in rows)
    assert report["statistics"]["independent_initial_roots"] == 1
    paths = [args.output_dir / (STEM + ext) for ext in (".pdf", ".svg")]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25))
    original, canonical, truth = "#35679C", "#CB6B34", "#687D71"
    ax = axes[0]
    xs, width = [0, 1], .32
    for offset, key, label, color in [(-width/2, "original_forecast_weighted_mse", "Original", original),
                                       (width/2, "canonical_forecast_weighted_mse", "Canonical", canonical)]:
        bars = ax.bar([x + offset for x in xs], [r[key] for r in rows], width,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(xs, ["A: Clean mean", "B: Canonical mean"])
    ax.set_ylabel("Terminal feature MSE (lower is better)")
    ax.set_ylim(0, 1.95)
    ax.set_title("(a) Both terminal forecasts improve", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    rewards = {r["candidate"]: r["cumulative_native_reward"] for r in rows}
    selectors = ["original", "canonical", "true_future"]
    selected = [report["cost_gap_decomposition"][s]["selected_candidate"] for s in selectors]
    values = [rewards[name] for name in selected]
    bars = ax.bar(range(3), values, width=.60, color=[original, canonical, truth])
    ax.bar_label(bars, labels=[f"{v:.2f}\nselects {'A' if name == rows[0]['candidate'] else 'B'}"
                             for v, name in zip(values, selected)], padding=4, fontsize=8)
    ax.set_xticks(range(3), ["Original\nforecast", "Canonical\nforecast", "True future\nfeatures"])
    ax.set_ylabel("Selected real 30-control return")
    ax.set_ylim(0, 120)
    ax.set_title("(b) The same score prefers lower utility", loc="left", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E7EA", linewidth=.6)
    fig.subplots_adjust(left=.085, right=.985, bottom=.24, top=.88, wspace=.37)
    fig.text(.5, .025, "One shared query; two post hoc selected means; fixed 30-control open-loop outcomes.\n"
             "True-future scoring is restricted to this bank, not a full oracle planner. No population inference.",
             ha="center", va="bottom", fontsize=7.5, color="#434C55")
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": "rendered_existing_results", "paths": [str(p) for p in paths],
                      "model_calls": 0, "environment_steps": 0, "api_calls": 0}))


if __name__ == "__main__":
    main()
