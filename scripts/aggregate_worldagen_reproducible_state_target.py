#!/usr/bin/env python3
"""Aggregate RNG-persisted state-target replay reproducibility audits."""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sources = []
    total_rows = 0
    exact_rows = 0
    initial_matches = 0
    qpos_max = 0.0
    qvel_max = 0.0
    eef_max = 0.0
    progress_max = 0.0
    contact_exact = 0
    image_exact = 0
    all_exact = 0
    task_seed_pairs = set()
    for raw in args.summaries:
        path = Path(raw)
        data = json.loads(path.read_text())
        rows = data.get("rows", [])
        total_rows += len(rows)
        exact_rows += sum(bool(r.get("match", {}).get("all_exact")) for r in rows)
        initial_matches += sum(
            r.get("source_initial_sim_sha256") == r.get("first_initial_sim_hash")
            for r in rows
        )
        qpos_max = max(qpos_max, *(r.get("match", {}).get("qpos_max_abs", 0.0) for r in rows))
        qvel_max = max(qvel_max, *(r.get("match", {}).get("qvel_max_abs", 0.0) for r in rows))
        eef_max = max(eef_max, *(r.get("match", {}).get("eef_abs", 0.0) for r in rows))
        progress_max = max(progress_max, *(r.get("match", {}).get("true_progress_abs", 0.0) for r in rows))
        contact_exact += sum(bool(r.get("match", {}).get("ncon_equal")) for r in rows)
        image_exact += sum(bool(r.get("match", {}).get("image_equal")) for r in rows)
        all_exact += sum(bool(r.get("match", {}).get("all_exact")) for r in rows)
        for r in rows:
            task_seed_pairs.add((data.get("seed"), r.get("task_id")))
        sources.append({
            "source": str(path),
            "seed": data.get("seed"),
            "tasks": data.get("task_ids"),
            "rows": len(rows),
            "all_replays_exact": bool(data.get("all_replays_exact")),
        })

    result = {
        "status": "completed_reproducible_state_target_aggregate",
        "sources": sources,
        "seed_count": len({s["seed"] for s in sources}),
        "task_seed_pairs": len(task_seed_pairs),
        "candidate_rows": total_rows,
        "replay_episodes": total_rows * 2,
        "all_replays_exact": exact_rows == total_rows and all(s["all_replays_exact"] for s in sources),
        "initial_sim_hash_matches": f"{initial_matches}/{total_rows}",
        "qpos_qvel_contact_image_eef_progress_exact": f"{all_exact}/{total_rows}",
        "contact_exact": f"{contact_exact}/{total_rows}",
        "image_exact": f"{image_exact}/{total_rows}",
        "max_qpos_abs_diff": qpos_max,
        "max_qvel_abs_diff": qvel_max,
        "max_eef_abs_diff": eef_max,
        "max_true_progress_abs_diff": progress_max,
        "model_inference": 0,
        "training_updates": 0,
        "api_requests": 0,
        "scope_limits": [
            "This is a replay reproducibility validation, not a Goal effect estimate.",
            "The protocol persists Python/NumPy/Torch/CUDA RNG states and compares fresh replays.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
