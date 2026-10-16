#!/usr/bin/env python3
"""Aggregate held-out forecast error under the fixed-action world cut."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_forecast_v0"
OUT = RUN / "summary.json"


def ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RUN.glob("window_*.json"))]
    if len(rows) != 7:
        raise SystemExit("need seven forecast windows")

    def a(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=float)

    native = a("native_task_a_world_forecast_mse")
    cut = a("cut_task_a_world_forecast_mse")
    zero = a("zero_context_world_forecast_mse")
    delta = cut - native
    swap = a("native_task_b_world_forecast_mse") - native
    result = {
        "status": "completed_alphabrain_fixed_action_forecast_gate",
        "cut_mode": "fixed_action_world",
        "windows": len(rows),
        "episodes": sorted(r["episode"] for r in rows),
        "native_task_a_world_forecast_mse_mean": float(native.mean()),
        "native_task_a_world_forecast_mse_bootstrap_ci95": ci(native, 7101),
        "cut_task_a_world_forecast_mse_mean": float(cut.mean()),
        "cut_task_a_world_forecast_mse_bootstrap_ci95": ci(cut, 7102),
        "zero_context_world_forecast_mse_mean": float(zero.mean()),
        "zero_context_world_forecast_mse_bootstrap_ci95": ci(zero, 7103),
        "repair_forecast_mse_delta_mean": float(delta.mean()),
        "repair_forecast_mse_delta_bootstrap_ci95": ci(delta, 7104),
        "repair_forecast_mse_delta_positive_fraction": float(np.mean(delta > 0)),
        "native_goal_swap_forecast_mse_delta_mean": float(swap.mean()),
        "native_goal_swap_forecast_mse_delta_bootstrap_ci95": ci(swap, 7105),
        "goal_cut_exact_zero_all": all(r["cut_future_world_delta_mean"] == 0 for r in rows),
        "rows": rows,
        "finding": "The fixed-action world cut preserves the recorded action carrier while removing text from action/value/world carriers. Its held-out forecast cost is measured against the same observed +16-frame target used by the original probe.",
        "scope": "Seven real image/state windows, fixed recorded actions, two-step sampler, world branch repair only; no environment rollout.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("windows", "episodes", "native_task_a_world_forecast_mse_mean", "cut_task_a_world_forecast_mse_mean", "zero_context_world_forecast_mse_mean", "repair_forecast_mse_delta_mean", "repair_forecast_mse_delta_bootstrap_ci95", "repair_forecast_mse_delta_positive_fraction", "goal_cut_exact_zero_all")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
