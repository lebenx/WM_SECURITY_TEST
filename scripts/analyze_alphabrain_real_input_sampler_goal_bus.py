#!/usr/bin/env python3
"""Aggregate the fixed-action two-step Cosmos sampler probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0"
OUT = RUN / "summary.json"


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RUN.glob("window_*.json"))]
    if len(rows) < 3 or any(r.get("status") != "completed_alphabrain_real_input_sampler_goal_bus_gate" for r in rows):
        raise SystemExit("need at least three completed sampler windows")
    native = np.asarray([r["native_future_world_delta_mean"] for r in rows], dtype=float)
    cut = np.asarray([r["cut_future_world_delta_mean"] for r in rows], dtype=float)
    policy = np.asarray([r["cut_policy_delta_mean"] for r in rows], dtype=float)
    native_zero = np.asarray([r["native_zero_context_future_world_delta_mean"] for r in rows], dtype=float)
    cut_zero = np.asarray([r["cut_zero_context_future_world_delta_mean"] for r in rows], dtype=float)
    result = {
        "status": "completed_alphabrain_real_input_sampler_goal_bus_gate",
        "sampler": "CosmosPolicySampler two-step reverse-ODE probe",
        "windows": len(rows),
        "episodes": sorted(r["episode"] for r in rows),
        "task_pairs": sorted({(r["task_a"], r["task_b"]) for r in rows}),
        "runtime_backend": rows[0]["runtime_backend"],
        "action_source": "recorded_executed_action_chunk",
        "action_source_all_recorded": all(r.get("action_source") == "recorded_executed_action_chunk" for r in rows),
        "native_future_world_delta_mean": float(native.mean()),
        "native_future_world_delta_min": float(native.min()),
        "native_future_world_delta_bootstrap_ci95": bootstrap_ci(native, 2701),
        "native_future_world_all_nonzero": bool(np.all(native > 0)),
        "cut_future_world_delta_mean": float(cut.mean()),
        "cut_future_world_max": float(cut.max()),
        "cut_future_world_exact_zero_all": bool(np.all(cut == 0)),
        "cut_policy_delta_mean": float(policy.mean()),
        "cut_policy_delta_bootstrap_ci95": bootstrap_ci(policy, 2702),
        "cut_policy_all_nonzero": bool(np.all(policy > 0)),
        "native_zero_context_future_world_delta_mean": float(native_zero.mean()),
        "cut_zero_context_future_world_delta_mean": float(cut_zero.mean()),
        "cut_zero_context_future_world_exact_zero_all": bool(np.all(cut_zero == 0)),
        "windows_detail": rows,
        "scope": "Three real LIBERO image/state windows with observation and recorded action slots fixed across goals; two-step Cosmos reverse-ODE denoising; no environment rollout.",
        "script_sha256": hashlib.sha256((ROOT / "scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_goal_bus.py").read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("windows", "episodes", "native_future_world_delta_mean", "native_future_world_delta_min", "native_future_world_delta_bootstrap_ci95", "cut_future_world_delta_mean", "cut_future_world_max", "cut_policy_delta_mean", "cut_policy_delta_bootstrap_ci95", "native_zero_context_future_world_delta_mean")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
