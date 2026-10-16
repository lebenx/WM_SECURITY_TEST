#!/usr/bin/env python3
"""Aggregate the short real-input AlphaBrain goal-bus windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alphabrain_cosmos_policy_real_input_goal_bus_v0"
OUT = RUN / "summary.json"


def bootstrap_ci(values: np.ndarray, seed: int = 1701, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    paths = sorted(RUN.glob("window_*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    if not rows or any(r.get("status") != "completed_alphabrain_real_input_goal_bus_gate" for r in rows):
        raise SystemExit("need completed window_*.json records")
    native = np.asarray([r["native_future_world_delta_mean"] for r in rows], dtype=float)
    cut = np.asarray([r["cut_future_world_delta_mean"] for r in rows], dtype=float)
    policy = np.asarray([r["cut_policy_delta_mean"] for r in rows], dtype=float)
    native_zero = np.asarray([r["native_zero_context_future_world_delta_mean"] for r in rows], dtype=float)
    cut_zero = np.asarray([r["cut_zero_context_future_world_delta_mean"] for r in rows], dtype=float)
    by_pair = {}
    for r in rows:
        pair = f'{r["task_a"]} || {r["task_b"]}'
        by_pair.setdefault(pair, []).append(r)
    task_pair_summary = []
    for pair, group in sorted(by_pair.items()):
        g_native = np.asarray([r["native_future_world_delta_mean"] for r in group], dtype=float)
        g_cut = np.asarray([r["cut_future_world_delta_mean"] for r in group], dtype=float)
        g_policy = np.asarray([r["cut_policy_delta_mean"] for r in group], dtype=float)
        task_pair_summary.append({
            "task_pair": pair,
            "windows": len(group),
            "episodes": sorted(r["episode"] for r in group),
            "native_future_world_delta_mean": float(g_native.mean()),
            "native_future_world_delta_min": float(g_native.min()),
            "cut_future_world_delta_max": float(g_cut.max()),
            "cut_policy_delta_mean": float(g_policy.mean()),
            "native_nonzero_all": bool(np.all(g_native > 0)),
            "cut_world_exact_zero_all": bool(np.all(g_cut == 0)),
            "cut_policy_nonzero_all": bool(np.all(g_policy > 0)),
        })
    result = {
        "status": "completed_alphabrain_real_input_goal_bus_gate",
        "windows": len(rows),
        "episodes": sorted(r["episode"] for r in rows),
        "frames": [r["frame"] for r in rows],
        "data_files": sorted({r["data_file"] for r in rows}),
        "task_pairs": sorted({(r["task_a"], r["task_b"]) for r in rows}),
        "task_pair_count": len({(r["task_a"], r["task_b"]) for r in rows}),
        "task_pair_summary": task_pair_summary,
        "runtime_backend": rows[0]["runtime_backend"],
        "action_source": "recorded_executed_action_chunk",
        "action_source_all_recorded": bool(all(r.get("action_source") == "recorded_executed_action_chunk" for r in rows)),
        "action_chunk_frame_ranges": [[r["action_chunk_frame_start"], r["action_chunk_frame_end"]] for r in rows],
        "vae_source": rows[0]["vae_source"],
        "vae_sha256": __import__("hashlib").sha256((ROOT / rows[0]["vae_source"]).read_bytes()).hexdigest(),
        "native_future_world_delta_mean": float(native.mean()),
        "native_future_world_delta_min": float(native.min()),
        "native_future_world_delta_bootstrap_ci95": bootstrap_ci(native),
        "native_future_world_all_nonzero": bool(np.all(native > 0)),
        "cut_future_world_delta_mean": float(cut.mean()),
        "cut_future_world_max": float(cut.max()),
        "cut_future_world_exact_zero_all": bool(np.all(cut == 0)),
        "cut_policy_delta_mean": float(policy.mean()),
        "cut_policy_delta_bootstrap_ci95": bootstrap_ci(policy, seed=1702),
        "cut_policy_all_nonzero": bool(np.all(policy > 0)),
        "native_zero_context_future_world_delta_mean": float(native_zero.mean()),
        "cut_zero_context_future_world_delta_mean": float(cut_zero.mean()),
        "cut_zero_context_future_world_exact_zero_all": bool(np.all(cut_zero == 0)),
        "windows_detail": rows,
        "scope": "Seven real LIBERO image/state windows from seven episodes across three public parquet files and all five task indices present in those files; each uses the recorded 16-step executed action chunk held fixed across goals, with one DiT forward per goal; no environment rollout.",
        "vae_provenance": {
            "repository": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "file": "vae/diffusion_pytorch_model.safetensors",
            "config_latent_channels": 16,
            "config_temporal_layout": "1 + (T - 1) // 4",
            "note": "Public Wan2.1 VAE has the same file size and latent mean/std configuration used by the gated Cosmos wrapper; the gated NVIDIA file itself was not downloaded.",
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["windows", "episodes", "native_future_world_delta_mean", "native_future_world_delta_min", "cut_future_world_delta_mean", "cut_future_world_max", "cut_policy_delta_mean", "native_zero_context_future_world_delta_mean", "cut_zero_context_future_world_delta_mean"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
