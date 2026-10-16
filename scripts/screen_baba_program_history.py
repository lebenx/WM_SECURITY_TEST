"""Evaluate published Alice programs on upstream training-map solution traces.

This is an offline diagnostic, not an Alice training run or a task-solving run.
All loading, serialization, sandboxing and prediction comparisons use upstream
code unchanged. Published solution traces are NOT the learner's online history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/BabaInWonderland"
DEMO = ROOT / "vendor/BabaInWonderland_demo"
sys.path.insert(0, str(UPSTREAM))

from scripts.data_collect.evaluate_offline_dataset import _load_state_archive_inputs
from scripts.data_collect.data_coverage import archive_state_input_to_canonical_text
from src.data import Transition
from src.program_model.evaluator import ProgramEvaluator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-pair-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    splits_path = UPSTREAM / "configs/custom_map_splits_dot_test.yaml"
    splits = yaml.safe_load(splits_path.read_text())["original"]
    maps = splits["all_train"]
    assert len(maps) == len(set(maps)) == 32
    assert not set(maps) & set(splits["all_test"])
    config_path = UPSTREAM / "configs/experiment_config_online.yaml"
    aliases = yaml.safe_load(config_path.read_text())["serialization"]["word_aliases"]
    provenance = {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "demo_bundle_sha256": digest(DEMO / "bundle.zip"),
        "script_sha256": digest(Path(__file__)),
        "split_sha256": digest(splits_path),
        "serialization_config_sha256": digest(config_path),
        "source": "Published demonstration programs and archived solution trajectories.",
        "split": "original/all_train",
        "maps": maps,
        "excluded_maps": splits["all_test"],
        "upstream_code_modified": False,
        "sandbox": "Upstream defaults; no additional timeout/line limits.",
        "evaluation_workers": args.workers,
        "alias_use": "Native observation remapping only, reproducing each published program's input interface.",
        "api_requests": 0,
        "gpu_work": False,
        "new_environment_interactions": 0,
        "new_training_runs": 0,
        "autonomous_task_episodes": 0,
        "final_pair_only": args.final_pair_only,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw_by_map = {}
    rows_by_map = {}
    source_hashes = {}
    for map_name in maps:
        state_path = UPSTREAM / "test_dataset/solution" / f"{map_name}_states.npz"
        row_path = UPSTREAM / "test_dataset/solution" / f"{map_name}_transitions.jsonl"
        raw_by_map[map_name] = _load_state_archive_inputs(state_path)
        rows_by_map[map_name] = [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
        source_hashes[map_name] = {"states_sha256": digest(state_path), "transitions_sha256": digest(row_path)}
    provenance["datasets"] = source_hashes
    save(args.output / "provenance.json", provenance)

    records_path = args.output / "programs.jsonl"
    variant_summaries = {}
    prediction_count = 0
    for variant in ["default", "wonderland"]:
        native_aliases = aliases if variant == "wonderland" else None
        transitions = []
        metadata = []
        for map_name in maps:
            states = [archive_state_input_to_canonical_text(raw, word_aliases=native_aliases)
                      for raw in raw_by_map[map_name]]
            for row in rows_by_map[map_name]:
                transitions.append(Transition(
                    state=states[row["state_index"]], action=row["action"],
                    next_state=states[row["next_state_index"]],
                    reward=float(row.get("reward", 0)), done=bool(row.get("done", False)),
                ))
                metadata.append({"map": map_name, "transition_index": row["transition_index"], "action": row["action"]})
        save(args.output / f"{variant}_transitions.json", metadata)
        paths = sorted((DEMO / "programs" / variant).glob("v*.py"))
        if args.final_pair_only:
            paths = paths[-2:]
        evaluator = ProgramEvaluator(program_eval_workers=args.workers)
        rows = []
        correct_vectors = []
        try:
            for path in paths:
                step_start = time.monotonic()
                evaluated = evaluator.evaluate_source(path.read_text(), transitions)
                assert not evaluated.compile_errors, (path, evaluated.compile_errors)
                assert len(evaluated.records) == len(transitions)
                correct = np.asarray([record.is_correct for record in evaluated.records], dtype=bool)
                assert int(correct.sum()) == evaluated.correct_count
                correct_vectors.append(correct)
                row = {
                    "variant": variant, "version": path.stem, "program_sha256": digest(path),
                    "correct": evaluated.correct_count, "total": evaluated.total_count,
                    "accuracy": evaluated.accuracy, "runtime_errors": evaluated.runtime_error_count,
                    "elapsed_s": time.monotonic() - step_start,
                    "by_map": {m: {"correct": sum(bool(c) for c, meta in zip(correct, metadata) if meta["map"] == m),
                                   "total": len(rows_by_map[m])} for m in maps},
                }
                rows.append(row)
                with records_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                prediction_count += len(transitions)
                print(json.dumps({k: row[k] for k in ["variant", "version", "correct", "total", "runtime_errors", "elapsed_s"]}), flush=True)
        finally:
            evaluator.close()
        matrix = np.stack(correct_vectors)
        np.savez_compressed(args.output / f"{variant}_correctness.npz", correct=matrix,
                            versions=np.asarray([p.stem for p in paths]))
        adjacent = []
        for i in range(1, len(paths)):
            gains = ~matrix[i-1] & matrix[i]
            losses = matrix[i-1] & ~matrix[i]
            adjacent.append({"before": paths[i-1].stem, "after": paths[i].stem,
                             "gains": int(gains.sum()), "losses": int(losses.sum()),
                             "retained": int((matrix[i-1] & matrix[i]).sum())})
        variant_summaries[variant] = {
            "program_count": len(paths), "transition_count": len(transitions),
            "final_correct": int(matrix[-1].sum()), "final_accuracy": float(matrix[-1].mean()),
            "ever_correct": int(matrix.any(axis=0).sum()),
            "previously_correct_final_wrong": int((matrix.any(axis=0) & ~matrix[-1]).sum()),
            "runtime_errors_total": sum(row["runtime_errors"] for row in rows),
            "adjacent_revision_deltas": adjacent,
            "adjacent_pairs_with_both_gains_and_losses": sum(row["gains"] > 0 and row["losses"] > 0 for row in adjacent),
            "note": "Solution-trace regressions do not imply regressions on the actual protected online history.",
        }
    summary = {
        "status": "completed", "variants": variant_summaries,
        "prediction_evaluations": prediction_count, "unique_archived_physical_transitions": len(transitions),
        "unique_training_maps": len(maps), "wall_time_s": time.monotonic()-started,
        "api_requests": 0, "gpu_work": False, "new_environment_interactions": 0,
        "new_training_runs": 0, "autonomous_task_episodes": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Programs are published versions from author demonstrations, not our independently trained replicas.",
            "Archived solution traces differ from the original online protected sets.",
            "Neither a rejected-update attack nor an exploration effect is established by this matrix.",
            "Train-map diagnostics are not held-out benchmark or task-completion results.",
        ],
    }
    save(args.output / "summary.json", summary)
    print(json.dumps({"status": "completed", "prediction_evaluations": prediction_count, "wall_time_s": summary["wall_time_s"]}), flush=True)


if __name__ == "__main__":
    main()
