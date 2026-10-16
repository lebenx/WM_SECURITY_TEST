#!/usr/bin/env python3
"""Extract simulator state/contact targets for the frozen WorldAgen candidates.

This does not run the model.  It replays the same first three actions used by
the aligned forecast audit and stores the simulator qpos/qvel/contact count as
an additional physical target for bounded downstream probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def reset_existing_env(env, init_state, seed):
    """Reset the wrapper/controller to the benchmark init without new renderer."""
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(7, dtype=np.float32))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-roots", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--suite", default="libero_10")
    args = ap.parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    rows = []
    expected_eef = []
    qpos, qvel, ncon = [], [], []
    for root in args.summary_roots:
        summary = json.loads((root / "summary.json").read_text())
        seed = int(summary["seed"])
        for row in summary["rows"]:
            task_id = int(row["task_id"])
            task = suite.get_task(task_id)
            payload = torch.load(root / f"task_{task_id}_forecasts.pt", map_location="cpu", weights_only=False)
            candidates = payload["candidates"].float()
            env, _ = ranking.make_env(task, args.suite, seed)
            init_file = ranking.libero.LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
            init_state = torch.load(init_file, map_location="cpu", weights_only=False)[0]
            try:
                # Reuse the renderer/controller but reset the full benchmark state
                # before each fixed candidate replay.
                for candidate_idx, candidate in enumerate(candidates):
                    obs = reset_existing_env(env, init_state, seed)
                    initial_hash = digest(env.env.sim.get_state().flatten())
                    if initial_hash != row["initial_sim_sha256"]:
                        raise AssertionError(f"initial state mismatch seed={seed} task={task_id}")
                    result = ranking.replay_once(env, obs, candidate[:3])
                    state = env.env.sim.get_state()
                    qpos.append(np.asarray(state.qpos, dtype=np.float32).copy())
                    qvel.append(np.asarray(state.qvel, dtype=np.float32).copy())
                    ncon.append(int(getattr(env.env.sim.data, "ncon", 0)))
                    rows.append({
                        "seed": seed, "task_id": task_id, "candidate_idx": candidate_idx,
                        "environment_steps": int(result["environment_steps"]),
                        "eef_approach_progress": float(result["eef_approach_progress"]),
                        "target_after_steps": 3,
                    })
                    expected_eef.append(float(row["candidate_replays"][candidate_idx]["eef_approach_progress"]))
            finally:
                env.close()
    max_eef_mismatch = max(abs(float(a["eef_approach_progress"]) - b)
                           for a, b in zip(rows, expected_eef))
    if max_eef_mismatch > 1e-7:
        invalid = {
            "status": "invalidated_not_used",
            "reason": "reconstructed replay did not reproduce the stored candidate utility",
            "max_eef_progress_mismatch": max_eef_mismatch,
            "next_requirement": "Persist controller/global RNG state during the original replay before extracting state/contact targets.",
            "model_inference": False,
            "api_requests": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(invalid, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(invalid))
        return
    # Tasks have different MuJoCo state dimensions.  NaN-padding retains the
    # exact per-task vectors without pretending coordinates are comparable.
    qpos_width = max(len(x) for x in qpos)
    qvel_width = max(len(x) for x in qvel)
    qpos = np.stack([np.pad(x, (0, qpos_width-len(x)), constant_values=np.nan) for x in qpos])
    qvel = np.stack([np.pad(x, (0, qvel_width-len(x)), constant_values=np.nan) for x in qvel])
    ncon = np.asarray(ncon, dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), qpos=qpos, qvel=qvel, ncon=ncon)
    metadata = {
        "status": "completed_physical_target_extraction",
        "summary_roots": [str(x) for x in args.summary_roots],
        "rows": rows,
        "arrays": {"qpos": list(qpos.shape), "qvel": list(qvel.shape), "ncon": list(ncon.shape),
                   "qpos_padding": "NaN to max task width", "qvel_padding": "NaN to max task width"},
        "target_after_steps": 3,
        "same_initial_state_verified": True,
        "model_inference": False,
        "training_updates": 0,
        "api_requests": 0,
        "scope_limits": [
            "Simulator states are physical targets for a downstream probe; they do not decode WorldAgen image latents by themselves.",
            "The extraction replays fixed candidates and does not change ranking or utility definitions.",
        ],
    }
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": metadata["status"], "rows": len(rows), "qpos": list(qpos.shape), "qvel": list(qvel.shape)}))


if __name__ == "__main__":
    main()
