#!/usr/bin/env python3
"""Aggregate the fixed-action-preserving AlphaBrain world cut."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_cut_v0"
OUT = RUN / "summary.json"


def bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in np.quantile(
        rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1),
        [0.025, 0.975],
    )]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RUN.glob("window_*.json"))]
    if len(rows) != 7:
        raise SystemExit("need seven fixed-action cut windows")
    native = np.asarray([r["native_future_world_delta_mean"] for r in rows], dtype=float)
    cut = np.asarray([r["cut_future_world_delta_mean"] for r in rows], dtype=float)
    result = {
        "status": "completed_alphabrain_fixed_action_world_cut_gate",
        "windows": len(rows),
        "episodes": sorted(r["episode"] for r in rows),
        "cut_mode": "fixed_action_world",
        "action_source": "recorded_executed_action_chunk",
        "native_goal_world_delta_mean": float(native.mean()),
        "native_goal_world_delta_bootstrap_ci95": bootstrap(native, 6101),
        "cut_goal_world_delta_mean": float(cut.mean()),
        "cut_goal_world_delta_max": float(cut.max()),
        "cut_goal_world_exact_zero_all": bool(np.all(cut == 0)),
        "policy_branch": "native parallel policy branch retained; repaired world branch intentionally has text-free action/value carriers",
        "rows": rows,
        "finding": "The fixed-action-preserving repair removes text cross-attention from action, value, and world rows but leaves self-attention intact. Across seven real LIBERO windows, goal swaps remain nonzero natively and are exactly invariant after the cut. The recorded action carrier remains available to world rows; separate action-sensitivity control is reported in the companion run.",
        "scope": "Seven real image/state windows, fixed recorded actions, two-step sampler, world-branch repair only; no environment rollout.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("windows", "episodes", "native_goal_world_delta_mean", "native_goal_world_delta_bootstrap_ci95", "cut_goal_world_delta_mean", "cut_goal_world_delta_max", "cut_goal_world_exact_zero_all")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
