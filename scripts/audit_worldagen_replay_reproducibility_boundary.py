#!/usr/bin/env python3
"""Audit within-run replay determinism versus post-hoc reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repeat_summary", type=Path)
    ap.add_argument("invalidated_attempt", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    repeat = json.loads(args.repeat_summary.read_text())
    invalid = json.loads(args.invalidated_attempt.read_text())
    spans = []
    repeat_counts = []
    for row in repeat["rows"]:
        for condition, replay in row["replays"].items():
            values = replay.get("repeat_results", [])
            repeat_counts.append(int(replay.get("replay_count", len(values))))
            for key in ["eef_approach_progress", "true_progress_delta", "eef_approach_distance_final"]:
                xs = [float(x[key]) for x in values]
                if xs:
                    spans.append(max(xs) - min(xs))
    result = {
        "status": "completed_replay_reproducibility_boundary_audit",
        "repeat_summary": str(args.repeat_summary),
        "invalidated_attempt": str(args.invalidated_attempt),
        "within_original_process": {
            "replay_counts": sorted(set(repeat_counts)),
            "max_numeric_span": float(max(spans)),
            "all_repeats_exact": bool(max(spans) == 0.0),
            "scope": "fresh-environment repeats inside the original ranking process",
        },
        "posthoc_reconstruction": {
            "status": invalid["status"],
            "max_eef_progress_mismatch": float(invalid["max_observed_eef_progress_mismatch"]),
            "scope": "new process with matching seed, candidate actions, and initial simulator hash",
        },
        "interpretation": "The saved decision results are internally deterministic under the original process, but the replay protocol is not externally reconstructible without persisting controller/global RNG state. This limits exact post-hoc qpos/qvel/contact target reconstruction; it does not invalidate the saved within-run counterfactual comparison.",
        "next_requirement": "Persist Python, NumPy, Torch, MuJoCo, and controller RNG/state snapshots in any new state-target experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
