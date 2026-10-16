#!/usr/bin/env python3
"""Aggregate AlphaBrain fixed-action sampler forecast-error probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_forecast_v0"
OUT = RUN / "summary.json"


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RUN.glob("window_*.json"))]
    expected = "completed_alphabrain_real_input_sampler_forecast_probe"
    if len(rows) < 7 or any(r.get("status") != expected for r in rows):
        raise SystemExit("need seven completed forecast windows")

    def arr(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=float)

    native_a = arr("native_task_a_world_forecast_mse")
    native_b = arr("native_task_b_world_forecast_mse")
    cut_a = arr("cut_task_a_world_forecast_mse")
    cut_b = arr("cut_task_b_world_forecast_mse")
    zero = arr("zero_context_world_forecast_mse")
    native_a_rel = arr("native_task_a_world_forecast_relative_l2")
    native_b_rel = arr("native_task_b_world_forecast_relative_l2")
    cut_a_rel = arr("cut_task_a_world_forecast_relative_l2")
    cut_b_rel = arr("cut_task_b_world_forecast_relative_l2")
    zero_rel = arr("zero_context_world_forecast_relative_l2")
    repair_delta = cut_a - native_a
    swap_delta = native_b - native_a

    result = {
        "status": "completed_alphabrain_real_input_sampler_forecast_aggregate",
        "windows": len(rows),
        "episodes": sorted(r["episode"] for r in rows),
        "runtime_backend": rows[0]["runtime_backend"],
        "action_source": "recorded_executed_action_chunk",
        "future_horizon_frames": 16,
        "target": "same-episode observed frame at current frame + 16, encoded by the same public VAE",
        "native_task_a_world_forecast_mse_mean": float(native_a.mean()),
        "native_task_a_world_forecast_mse_bootstrap_ci95": bootstrap_ci(native_a, 4101),
        "native_task_b_world_forecast_mse_mean": float(native_b.mean()),
        "native_task_b_world_forecast_mse_bootstrap_ci95": bootstrap_ci(native_b, 4102),
        "cut_task_a_world_forecast_mse_mean": float(cut_a.mean()),
        "cut_task_a_world_forecast_mse_bootstrap_ci95": bootstrap_ci(cut_a, 4103),
        "cut_task_b_world_forecast_mse_mean": float(cut_b.mean()),
        "cut_task_b_world_forecast_mse_bootstrap_ci95": bootstrap_ci(cut_b, 4104),
        "zero_context_world_forecast_mse_mean": float(zero.mean()),
        "zero_context_world_forecast_mse_bootstrap_ci95": bootstrap_ci(zero, 4107),
        "repair_forecast_mse_delta_mean": float(repair_delta.mean()),
        "repair_forecast_mse_delta_bootstrap_ci95": bootstrap_ci(repair_delta, 4105),
        "repair_forecast_mse_delta_positive_fraction": float(np.mean(repair_delta > 0)),
        "native_goal_swap_forecast_mse_delta_mean": float(swap_delta.mean()),
        "native_goal_swap_forecast_mse_delta_bootstrap_ci95": bootstrap_ci(swap_delta, 4106),
        "native_task_a_world_forecast_relative_l2_mean": float(native_a_rel.mean()),
        "native_task_b_world_forecast_relative_l2_mean": float(native_b_rel.mean()),
        "cut_task_a_world_forecast_relative_l2_mean": float(cut_a_rel.mean()),
        "cut_task_b_world_forecast_relative_l2_mean": float(cut_b_rel.mean()),
        "zero_context_world_forecast_relative_l2_mean": float(zero_rel.mean()),
        "native_goal_swap_delta_all_nonzero": all(r["native_future_world_delta_mean"] > 0 for r in rows),
        "cut_goal_invariance_exact_zero_all": all(r["cut_future_world_delta_mean"] == 0 for r in rows),
        "windows_detail": rows,
        "scope": "Seven real LIBERO image/state windows; fixed recorded action chunks; two-step sampler; observed future latent target held out from denoising; no environment rollout.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "windows", "episodes", "native_task_a_world_forecast_mse_mean",
        "native_task_b_world_forecast_mse_mean", "cut_task_a_world_forecast_mse_mean",
        "repair_forecast_mse_delta_mean", "repair_forecast_mse_delta_bootstrap_ci95",
        "repair_forecast_mse_delta_positive_fraction", "native_goal_swap_forecast_mse_delta_mean",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
