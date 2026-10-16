#!/usr/bin/env python3
"""Pilot a replay protocol that persists and restores all local RNG sources.

This consumes frozen candidates from the existing aligned sweep and does not
run WorldAgen inference.  It is a reproducibility gate for future qpos/qvel/
contact targets, not a new attack or utility estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "LIBERO"))
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_worldagen_counterfactual_action_ranking as ranking  # noqa: E402


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def snapshot_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_digest(state):
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def one_replay(task, suite_name, seed, candidate):
    env, obs = ranking.make_env(task, suite_name, seed)
    try:
        initial_hash = digest(env.env.sim.get_state().flatten())
        result = ranking.replay_once(env, obs, candidate[:3])
        after = result["observations_after_steps"][3]
        sim = env.env.sim
        return {
            "initial_sim_hash": initial_hash,
            "qpos": np.asarray(sim.data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(sim.data.qvel, dtype=np.float64).copy(),
            "ncon": int(getattr(sim.data, "ncon", 0)),
            "eef_approach_progress": float(result["eef_approach_progress"]),
            "true_progress_delta": float(result["true_progress_delta"]),
            "environment_steps": int(result["environment_steps"]),
            "target_image_hash": digest(after["agentview_image"]),
        }
    finally:
        env.close()


def compare(a, b):
    return {
        "qpos_max_abs": float(np.max(np.abs(a["qpos"] - b["qpos"]))),
        "qvel_max_abs": float(np.max(np.abs(a["qvel"] - b["qvel"]))),
        "ncon_equal": bool(a["ncon"] == b["ncon"]),
        "eef_abs": float(abs(a["eef_approach_progress"] - b["eef_approach_progress"])),
        "true_progress_abs": float(abs(a["true_progress_delta"] - b["true_progress_delta"])),
        "image_equal": bool(a["target_image_hash"] == b["target_image_hash"]),
        "all_exact": bool(
            np.array_equal(a["qpos"], b["qpos"]) and np.array_equal(a["qvel"], b["qvel"]) and
            a["ncon"] == b["ncon"] and a["target_image_hash"] == b["target_image_hash"] and
            a["eef_approach_progress"] == b["eef_approach_progress"] and
            a["true_progress_delta"] == b["true_progress_delta"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-root", type=Path, required=True)
    ap.add_argument("--task-ids", default="0,2")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = [int(x) for x in args.task_ids.split(",") if x]
    rows = []
    state_arrays = []
    for task_id in tasks:
        task = suite.get_task(task_id)
        payload = torch.load(args.forecast_root / f"task_{task_id}_forecasts.pt", map_location="cpu", weights_only=False)
        candidates = payload["candidates"].float()
        summary = json.loads((args.forecast_root / "summary.json").read_text())
        source_row = next(x for x in summary["rows"] if int(x["task_id"]) == task_id)
        for candidate_idx, candidate in enumerate(candidates):
            state = snapshot_rng()
            first = one_replay(task, args.suite, args.seed, candidate)
            restore_rng(state)
            second = one_replay(task, args.suite, args.seed, candidate)
            match = compare(first, second)
            rows.append({
                "task_id": task_id, "candidate_idx": candidate_idx,
                "source_initial_sim_sha256": source_row["initial_sim_sha256"],
                "first_initial_sim_hash": first["initial_sim_hash"],
                "rng_snapshot_sha256": rng_digest(state),
                "match": match,
                "first": {k: v for k, v in first.items() if k not in {"qpos", "qvel"}},
            })
            state_arrays.append((task_id, candidate_idx, first["qpos"], first["qvel"], first["ncon"]))
    all_exact = all(x["match"]["all_exact"] for x in rows)
    result = {
        "status": "completed_reproducible_state_target_pilot" if all_exact else "failed_reproducible_state_target_pilot",
        "forecast_root": str(args.forecast_root), "task_ids": tasks, "seed": args.seed,
        "candidate_count": 8, "replay_steps": 3, "rows": rows,
        "all_replays_exact": all_exact,
        "model_inference": False, "training_updates": 0, "api_requests": 0,
        "scope_limits": [
            "This is a reproducibility gate; it does not compare Goal conditions or claim attack success.",
            "State targets are valid only for this fully persisted RNG/controller protocol.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if all_exact:
        qpos_width = max(len(x[2]) for x in state_arrays)
        qvel_width = max(len(x[3]) for x in state_arrays)
        qpos = np.stack([np.pad(x[2], (0, qpos_width-len(x[2])), constant_values=np.nan)
                         for x in state_arrays])
        qvel = np.stack([np.pad(x[3], (0, qvel_width-len(x[3])), constant_values=np.nan)
                         for x in state_arrays])
        ncon = np.asarray([x[4] for x in state_arrays], dtype=np.int32)
        np.savez_compressed(args.output.with_suffix(".npz"), qpos=qpos, qvel=qvel, ncon=ncon)
    print(json.dumps({"status": result["status"], "rows": len(rows), "all_replays_exact": all_exact}))


if __name__ == "__main__":
    main()
