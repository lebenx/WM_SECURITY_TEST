#!/usr/bin/env python3
"""Reuse the frozen native N2 task runner with method-specific VAE residency.

This is an execution-resource experiment, not an attack or a new planner.
No old runner, checkpoint, source manifest or result is modified. The reference
provenance is checked before reuse; full-episode parity includes every recorded
control, candidate score/selection, physical trace, simulator state and PNG.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import pilot_cosmos_policy_native_task_gate as native

ROOT = native.ROOT
DEFAULT_REFERENCE = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_v1/summary.json"


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def compare_episode(reference, actual):
    """Compare scientific fields exactly, excluding execution time/memory."""
    result_keys = ("suite", "best_of_n", "task_id", "init_id", "goal", "seed",
                   "environment_seed", "max_steps", "settle_steps", "initial_state_sha256",
                   "bddl_sha256", "init_file_sha256", "success_at_start", "success",
                   "terminal_done", "environment_steps", "action_chunks", "reward_sum",
                   "final_reward", "action_norm_mean")
    checks = {f"result_{key}": reference[key] == actual[key] for key in result_keys}
    checks["all_physical_trace_fields"] = records(reference["trace"]) == records(actual["trace"])
    old_chunks, new_chunks = records(reference["chunks"]), records(actual["chunks"])
    ignored = {"query_seconds", "cuda_allocated_bytes"}
    clean = lambda rows: [{k: v for k, v in row.items() if k not in ignored} for row in rows]
    checks["all_candidate_scores_selections_actions_and_future_rms"] = clean(old_chunks) == clean(new_chunks)
    for key in ("initial_image", "final_image"):
        checks[key] = native.digest(reference[key]) == native.digest(actual[key])
    old_states = Path(reference["trace"]).with_name(Path(reference["trace"]).stem + "_states.npz")
    new_states = Path(actual["trace"]).with_name(Path(actual["trace"]).stem + "_states.npz")
    with np.load(old_states) as before, np.load(new_states) as after:
        checks["state_array_names"] = set(before.files) == set(after.files)
        for key in before.files:
            checks[f"state_{key}"] = key in after.files and np.array_equal(before[key], after[key])
    return {"all_exact": all(checks.values()), "checks": checks,
            "reference_trace_sha256": native.digest(reference["trace"]),
            "actual_trace_sha256": native.digest(actual["trace"]),
            "reference_steps": reference["environment_steps"],
            "actual_steps": actual["environment_steps"],
            "scope": "One complete paired native N2 episode; not full upstream TE backend parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--task-ids", type=native.parse_ids, default=[0])
    parser.add_argument("--init-ids", type=native.parse_ids, default=[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.055)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cuda_memory_fraction <= 1:
        parser.error("allocator fraction must be in (0,1]")
    source = json.loads(args.reference_summary.read_text())
    if source["status"] != "completed" or source["protocol"]["best_of_n"] != 2:
        raise ValueError("Expected a completed native N2 reference")
    provenance = copy.deepcopy(source["provenance"])
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"), ("tokenizer", "tokenizer_sha256")):
        native.inference.verify_artifact(Path(provenance[path_key]), provenance[hash_key])
    native.inference.verify_artifact(Path(native.__file__), provenance["runner_sha256"])
    native.inference.verify_artifact(Path(native.inference.__file__), provenance["helper_sha256"])
    for relative, expected in provenance["source_sha256"].items():
        native.inference.verify_artifact(ROOT / relative, expected)
    for key in ("suite", "seed", "env_seed", "max_steps", "settle_steps", "best_of_n"):
        setattr(args, key, source["protocol"][key])
    args.video_stride = 0
    args.offload = True
    args.out_dir = args.out_dir.resolve()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    text_path = native.CKPT_DIR / "libero_t5_embeddings.pkl"
    stats_path = native.CKPT_DIR / "libero_dataset_statistics.json"
    native.inference.verify_artifact(text_path, provenance["text_embeddings_sha256"])
    native.inference.verify_artifact(stats_path, provenance["stats_sha256"])
    with text_path.open("rb") as handle:
        texts = pickle.load(handle)
    stats = native.inference.load_stats(stats_path)
    episodes = []
    for tid in args.task_ids:
        if tid >= suite.n_tasks:
            raise ValueError("Task index outside suite")
        task = suite.get_task(tid)
        states = suite.get_task_init_states(tid)
        for iid in args.init_ids:
            if iid >= len(states):
                raise ValueError("Initial state outside task")
            episodes.append({"task_id": tid, "init_id": iid, "goal": task.language})
    if not episodes or episodes[0]["task_id"] != 0 or episodes[0]["init_id"] != 0:
        raise ValueError("Begin with Task0/Init0 for the full-episode equivalence gate")
    import cosmos_policy_cuda_branch_offload as branch
    provenance["reference_runner_sha256"] = provenance["runner_sha256"]
    provenance["runner_sha256"] = native.digest(Path(__file__))
    provenance["branch_offload_helper_sha256"] = native.digest(Path(branch.__file__))
    provenance["parent_offload_helper_sha256"] = native.digest(ROOT / "scripts/cosmos_policy_cuda_offload.py")
    provenance["source_reference_summary_sha256"] = native.digest(args.reference_summary)
    provenance["execution_change"] = "Only encode/decode-specific VAE weight residency; original CUDA operators/dtypes"
    if args.preflight:
        print(json.dumps({"status": "cpu_preflight_passed", "episodes": episodes, "provenance": provenance}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        parser.error("Output must be a fresh directory")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    started = time.monotonic()
    summary = {"status": "loading", "condition": "clean", "provenance": provenance,
               "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "planned_episodes": episodes, "results": []}
    path = args.out_dir / "summary.json"
    native.atomic_json(path, summary)
    try:
        model, vae, sampler, rand = native.inference.load_model(device="cpu")
        model, vae, manager, metadata = branch.prepare(model, vae)
        provenance["execution_offload"] = metadata
        summary["status"] = "running"
        native.atomic_json(path, summary)
        for item in episodes:
            result = native.run_episode(model, vae, sampler, rand, suite.get_task(item["task_id"]),
                                        item["task_id"], item["init_id"], texts, stats, args, provenance)
            summary["results"].append(result)
            if len(summary["results"]) == 1:
                reference = next(row for row in source["results"] if row["task_id"] == 0 and row["init_id"] == 0)
                parity = compare_episode(reference, result)
                native.atomic_json(args.out_dir / "equivalence.json", parity)
                summary["full_episode_equivalence"] = parity
                if not parity["all_exact"]:
                    raise RuntimeError("Full-episode parity failed; do not extend cross-task gate")
            successes = sum(row["success"] for row in summary["results"])
            summary["aggregate"] = {"episodes": len(summary["results"]), "successes": successes,
                                    "success_rate": successes / len(summary["results"])}
            summary["elapsed_seconds"] = time.monotonic() - started
            native.atomic_json(path, summary)
            print(json.dumps({"completed": len(summary["results"]), "task_id": item["task_id"],
                              "init_id": item["init_id"], "success": result["success"],
                              "steps": result["environment_steps"]}), flush=True)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="error", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        native.atomic_json(path, summary)
    print(json.dumps({"summary": str(path), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
