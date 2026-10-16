#!/usr/bin/env python3
"""Plot the bounded AlphaBrain goal-free-context comparator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("seed 17\nframe 30", ROOT / "runs/alphabrain_goal_free_context_v1/summary.json"),
    ("seed 23\nframe 30", ROOT / "runs/alphabrain_goal_free_context_v2/summary.json"),
    ("seed 29\nframes 30/60/90", ROOT / "runs/alphabrain_goal_free_context_v3/summary.json"),
]


def main() -> None:
    labels, initial, learned = [], [], []
    for label, path in RUNS:
        result = json.loads(path.read_text(encoding="utf-8"))
        labels.append(label)
        initial.append(result["initial_zero_context_eval"]["mse_mean"])
        learned.append(result["learned_context_eval"]["mse_mean"])

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(x - width / 2, initial, width, label="zero context", color="#b8c4d6")
    ax.bar(x + width / 2, learned, width, label="learned shared context", color="#2f6f9f")
    for i, (a, b) in enumerate(zip(initial, learned)):
        ax.text(i + width / 2, b + 0.012, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i - width / 2, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out one-step world MSE")
    ax.set_title("Bounded goal-free context comparator\n(frozen AlphaBrain DiT/VAE; no task fine-tuning)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "goal_free_context_baseline.svg")
    fig.savefig(out / "goal_free_context_baseline.pdf")
    print(json.dumps({"outputs": [str(out / "goal_free_context_baseline.svg"), str(out / "goal_free_context_baseline.pdf")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
