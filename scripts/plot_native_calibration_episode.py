#!/usr/bin/env python3
"""Plot measured native episode returns and the earlier open-loop countercheck."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPISODE = ROOT / "runs/native_forecast_calibration_episode_v1"
PREFIX = ROOT / "runs/native_forecast_calibration_gate_v1"
OUT = ROOT / "reports/decision_leverage_paper"


def read_rewards(directory, arm, length):
    rows = [json.loads(line) for line in (directory / arm / "trace.jsonl").read_text().splitlines()]
    if len(rows) != length or [r["policy_step"] for r in rows] != list(range(1, length + 1)):
        raise ValueError("Incomplete or misaligned physical trajectory: " + str(directory / arm))
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise ValueError("Nonfinite physical reward")
    return np.cumsum(rewards)


def main():
    summary = json.loads((EPISODE / "summary.json").read_text())
    if summary["status"] != "completed_single_seed_full_native_episode_calibration":
        raise ValueError("Only a completed native episode can produce this figure")
    curves = {arm: read_rewards(EPISODE, arm, 99) for arm in ("raw", "calibrated")}
    open_loop = {arm: read_rewards(PREFIX, arm, 30) for arm in ("raw", "calibrated")}
    for branch in summary["branches"]:
        if not np.isclose(curves[branch["arm"]][-1], branch["utility"]["cumulative_return"], rtol=0, atol=1e-8):
            raise ValueError("Trace return disagrees with recorded episode outcome")
    delta = curves["calibrated"] - curves["raw"]
    open_delta = open_loop["calibrated"] - open_loop["raw"]
    if not np.array_equal(delta[:15], open_delta[:15]):
        raise ValueError("Native first execution prefix differs from earlier countercheck")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout="constrained")
    t = np.arange(1, 100)
    for arm, label, color in (("raw", "Original", "#286a9b"),
                              ("calibrated", "Fixed residual correction", "#b96728")):
        axes[0].plot(t, curves[arm], label=label, color=color, linewidth=2)
    axes[0].set_ylabel("Cumulative environment reward")
    axes[0].set_title("A. Complete native execution, same initial state")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].plot(t, delta, color="#286a9b", linewidth=2, label="Native replanning (99 controls)")
    axes[1].plot(np.arange(1, 31), open_delta, color="#b96728", linestyle="--",
                 linewidth=2, label="Saved first plan, open loop (30 controls)")
    axes[1].set_ylabel("Corrected minus original cumulative reward")
    axes[1].set_title("B. Execution protocol changes the comparison")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    for ax in axes:
        for step in range(15, 99, 15):
            ax.axvline(step + .5, color="#777777", alpha=.16, linewidth=.7)
        ax.set_xlabel("Executed policy controls (warmup excluded)")
        ax.set_xlim(1, 99)
        ax.grid(axis="y", alpha=.12)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / ("native_calibration_episode." + extension), dpi=180)
    plt.close(fig)
    sources = {
        "scope": "One previously inspected root3001; no population inference or attack claim. Initial-query error reduction is not full-episode accuracy.",
        "native_summary": str((EPISODE / "summary.json").relative_to(ROOT)),
        "native_traces": [str((EPISODE / arm / "trace.jsonl").relative_to(ROOT)) for arm in curves],
        "open_loop_traces": [str((PREFIX / arm / "trace.jsonl").relative_to(ROOT)) for arm in curves],
        "raw_episode_return": float(curves["raw"][-1]),
        "corrected_episode_return": float(curves["calibrated"][-1]),
        "episode_return_delta": float(delta[-1]),
        "open_loop_30_return_delta": float(open_delta[-1]),
        "model_calls": 0, "environment_controls": 0,
    }
    (OUT / "native_calibration_episode.sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
