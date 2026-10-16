#!/usr/bin/env python3
"""Plot the held-out AlphaBrain forecast tradeoff per real LIBERO window."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json"
out = ROOT / "reports/figures/alphabrain_forecast_tradeoff.svg"
d = json.loads(src.read_text())
rows = sorted(d.get("windows_detail", d["rows"]), key=lambda r: (r["episode"], r.get("frame", 0)))
x = np.arange(len(rows))
native_a = [r["native_task_a_world_forecast_mse"] for r in rows]
native_b = [r["native_task_b_world_forecast_mse"] for r in rows]
cut = [r["cut_task_a_world_forecast_mse"] for r in rows]
zero = [r["zero_context_world_forecast_mse"] for r in rows]

plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(x, native_a, "o-", label="native goal A", linewidth=1.8)
ax.plot(x, native_b, "s--", label="native legal goal B", linewidth=1.5)
ax.plot(x, cut, "D-", label="fixed-action world cut (A=B)", linewidth=1.8)
ax.plot(x, zero, "^-.", label="zero T5 context", linewidth=1.5)
ax.set_xticks(x, [f'{r["episode"]}:{r.get("frame", "")}' for r in rows], rotation=65, ha="right")
ax.set_xlabel("LIBERO episode")
ax.set_ylabel("held-out world latent MSE")
ax.set_title("AlphaBrain 21-window fixed-action forecast matrix")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, format="svg", metadata={"Source": str(src.relative_to(ROOT))})
print(out)
