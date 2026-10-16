#!/usr/bin/env python3
"""Plot the WorldAgen activation-mediation residual by Transformer layer."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__import__("os").environ.get(
    "WORLDAGEN_ACTIVATION_PATCH_PLOT_SOURCE",
    str(ROOT / "runs/worldagen_goal_bus_activation_patch_v1/summary.json"),
))
OUT = ROOT / "reports/figures/worldagen_activation_mediation"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    rows = data["aggregate_by_layer"]
    layers = [int(x) for x in rows]
    world = [rows[str(x)]["world_residual_fraction_mean"] for x in layers]
    policy = [rows[str(x)]["policy_residual_fraction_mean"] for x in layers]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(layers, world, marker="o", linewidth=2, label="world residual")
    ax.plot(layers, policy, marker="s", linewidth=2, label="policy residual")
    ax.axhline(1.0, color="0.55", linewidth=1, linestyle="--")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Transformer layer where action states are patched")
    ax.set_ylabel("distance to recorded-goal output / native swap")
    ax.set_title(f"Action-token activation mediation ({data['windows']} fixed windows)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUT.with_suffix('.svg')} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
