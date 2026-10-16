#!/usr/bin/env python3
"""Plot actual native traces; no model scores or new environment execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/dino_native_alias_pair_20261015")
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for arm, label, color in (("clean_identity_copy", "Clean", "#1764a3"),
                              ("canonical_request", "Canonical request", "#bf541d"),
                              ("noop", "Zero action", "#777777")):
        rows = [json.loads(line) for line in (args.run / arm / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == 99
        steps = [r["policy_step"] for r in rows]
        rewards = np.array([r["reward"] for r in rows])
        distance = [r["reach_distance"] for r in rows]
        for ax, values in zip(axes, (rewards, rewards.cumsum(), distance)):
            ax.plot(steps, values, color=color, label=label, linewidth=1.8)
    for ax, label in zip(axes, ("Actual step reward", "Cumulative actual reward", "Reach distance (m)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(loc="lower right", ncol=3, fontsize=9)
    axes[-1].set_xlabel("Policy control step (shared warmup excluded)")
    axes[-1].set_xlim(1, 99)
    fig.suptitle("DINO native CEM: one fixed seed4 pair\nLower cumulative reward does not imply worse terminal goal attainment", fontsize=12)
    for extension in ("svg", "png"):
        path = args.run / ("actual_utility." + extension)
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
