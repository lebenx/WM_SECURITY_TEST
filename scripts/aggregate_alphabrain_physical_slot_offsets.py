#!/usr/bin/env python3
"""Aggregate AlphaBrain future-output Goal effects across legal offsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(rows: list[dict], key: str, draws: int = 20000, seed: int = 20261012) -> dict:
    episodes = sorted({int(r["episode"]) for r in rows})
    values = np.asarray([float(r[key]) for r in rows])
    ep = np.asarray([values[[int(r["episode"]) == e for r in rows]].mean() for e in episodes])
    rng = np.random.default_rng(seed)
    boot = ep[rng.integers(0, len(ep), size=(draws, len(ep)))].mean(axis=1)
    return {
        "n_windows": int(values.size), "episode_count": len(episodes),
        "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "episode_cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "positive_fraction": float(np.mean(values > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--offsets", nargs="+", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(args.inputs) != len(args.offsets):
        raise SystemExit("inputs and offsets must have equal length")
    per_offset = {}
    for path, offset in zip(args.inputs, args.offsets):
        data = json.loads(Path(path).read_text())
        rows = data["rows"]
        per_offset[str(offset)] = {
            "source": str(path), "windows": len(rows),
            "native_goal_effect_abs_delta": {
                n: summary(rows, f"native_goal_{n}_abs_delta")
                for n in ("future_proprio", "future_wrist", "future_primary")
            },
            "native_goal_target_mse_delta": {
                n: summary(rows, f"native_goal_{n}_mse_delta")
                for n in ("future_proprio", "future_wrist", "future_primary")
            },
            "fixed_action_cut_exact_zero": all(
                all(float(r[f"cut_goal_{n}_abs_delta"]) == 0.0 for n in ("future_proprio", "future_wrist", "future_primary"))
                for r in rows
            ),
        }
    result = {
        "status": "completed_alphabrain_physical_slot_offset_audit",
        "offsets": args.offsets, "per_offset": per_offset,
        "interpretation": "Across legal offsets, Goal changes future physical-output slots under fixed state/action; target-error direction remains a separate estimand.",
        "scope_limits": [
            "Seven episode clusters are reused across offsets; intervals are episode-clustered within each offset.",
            "Future proprio is a model output slot compared with an observed +16-frame state target, not native task utility.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
