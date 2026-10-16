#!/usr/bin/env python3
"""Aggregate AlphaBrain legal task-offset sampler probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_offsets_v0"
SOURCE = ROOT / "runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0/summary.json"
OUT = OUT_DIR / "summary.json"


def cluster_ci(rows: list[dict], key: str, seed: int, draws: int = 10000) -> list[float]:
    episodes = sorted({int(r["episode"]) for r in rows})
    by_ep = {ep: [r for r in rows if int(r["episode"]) == ep] for ep in episodes}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        chosen = rng.choice(episodes, size=len(episodes), replace=True)
        sample = [r[key] for ep in chosen for r in by_ep[int(ep)]]
        vals.append(float(np.mean(sample)))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main() -> None:
    rows = []
    old = json.loads(SOURCE.read_text())
    for r in old["windows_detail"]:
        if int(r["episode"]) in {3, 6, 8}:
            rows.append(r)
    rows.extend(json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("episode_*.json")))
    if len(rows) != 9 or sorted(int(r["task_offset"]) for r in rows) != [1, 1, 1, 5, 5, 5, 9, 9, 9]:
        raise SystemExit("expected 3 episodes x offsets 1,5,9")
    native = np.asarray([r["native_future_world_delta_mean"] for r in rows], dtype=float)
    cut = np.asarray([r["cut_future_world_delta_mean"] for r in rows], dtype=float)
    policy = np.asarray([r["cut_policy_delta_mean"] for r in rows], dtype=float)
    by_offset = {}
    for off in (1, 5, 9):
        sub = [r for r in rows if int(r["task_offset"]) == off]
        by_offset[str(off)] = {
            "windows": len(sub),
            "native_world_delta_mean": float(np.mean([r["native_future_world_delta_mean"] for r in sub])),
            "native_world_all_nonzero": all(r["native_future_world_delta_mean"] > 0 for r in sub),
            "cut_world_exact_zero_all": all(r["cut_future_world_delta_mean"] == 0 for r in sub),
            "cut_policy_delta_mean": float(np.mean([r["cut_policy_delta_mean"] for r in sub])),
        }
    result = {
        "status": "completed_alphabrain_legal_offset_sampler_matrix",
        "windows": len(rows),
        "physical_episodes": [3, 6, 8],
        "task_offsets": [1, 5, 9],
        "offsets_are_clustered_by_episode": True,
        "native_world_delta_mean": float(native.mean()),
        "native_world_delta_cluster_bootstrap_ci95": cluster_ci(rows, "native_future_world_delta_mean", 5101),
        "native_world_all_nonzero": bool(np.all(native > 0)),
        "cut_world_delta_mean": float(cut.mean()),
        "cut_world_exact_zero_all": bool(np.all(cut == 0)),
        "cut_policy_delta_mean": float(policy.mean()),
        "cut_policy_cluster_bootstrap_ci95": cluster_ci(rows, "cut_policy_delta_mean", 5102),
        "by_offset": by_offset,
        "action_source_all_recorded": all(r.get("action_source") == "recorded_executed_action_chunk" for r in rows),
        "rows": rows,
        "scope": "Three real LIBERO image/state windows crossed with three legal task offsets; fixed recorded actions and two-step sampler; no environment rollout.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("windows", "physical_episodes", "task_offsets", "native_world_delta_mean", "native_world_delta_cluster_bootstrap_ci95", "cut_world_delta_mean", "cut_policy_delta_mean", "cut_policy_cluster_bootstrap_ci95", "by_offset")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
