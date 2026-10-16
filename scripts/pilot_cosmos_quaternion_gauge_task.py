#!/usr/bin/env python3
"""Eight preregistered native episodes, changing only the model's quaternion copy.

This is an observation-representation development screen, not a novelty claim.
The frozen native runner owns all simulation, action execution and success tests.
All native proposal/future/value calls see the same transformed observation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import cosmos_policy_native_inference as inference
import pilot_cosmos_policy_native_task_gate as base
from cosmos_counterfactual_replay import array_sha
from pilot_cosmos_policy_branch_offload_task_gate import compare_episode

ROOT = base.ROOT
CONFIG = ROOT / "configs/cosmos_quaternion_gauge_screen_v1.json"
OUT = ROOT / "runs/cosmos_quaternion_gauge_task_v1"
GOAL = "open the middle drawer of the cabinet"
CONDITIONS = ("clean", "sign_flip", "sign_flip_then_x_canonical", "clean_x_canonical")
PLAN = [(n, condition) for condition in CONDITIONS for n in (1, 2)]


def rotation_matrix(q):
    """Unit-quaternion polynomial, xyzw; intentionally never normalize inputs."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def canonical(q):
    """Choose positive first nonzero coordinate in fixed order x,y,z,w."""
    for value in q:
        if value != 0:
            return -q if value < 0 else q.copy()
    raise ValueError("Zero quaternion is not a valid orientation")


def transform_observation(obs, condition, stats):
    if condition not in CONDITIONS:
        raise ValueError("Unregistered condition")
    q = np.asarray(obs["robot0_eef_quat"])
    if q.shape != (4,) or q.dtype.kind != "f" or not np.isfinite(q).all():
        raise ValueError("Malformed xyzw quaternion")
    if abs(float(np.linalg.norm(q)) - 1) > 1e-6:
        raise ValueError("Expected raw unit quaternion; normalization is forbidden")
    before = {k: array_sha(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
    modeled = copy.deepcopy(obs)
    result = -q if condition in ("sign_flip", "sign_flip_then_x_canonical") else q.copy()
    if condition in ("sign_flip_then_x_canonical", "clean_x_canonical"):
        result = canonical(result)
    modeled["robot0_eef_quat"] = result
    after = {k: array_sha(v) for k, v in modeled.items() if isinstance(v, np.ndarray)}
    protected = {k: before[k] == after[k] for k in before if k != "robot0_eef_quat"}
    if not all(protected.values()) or before != {k: array_sha(v) for k, v in obs.items() if isinstance(v, np.ndarray)}:
        raise RuntimeError("Transformation modified a protected field or original observation")
    same_rotation = np.array_equal(rotation_matrix(q), rotation_matrix(result))
    same_norm = np.linalg.norm(q) == np.linalg.norm(result)
    if not same_rotation or not same_norm:
        raise RuntimeError("Transformation changed physical orientation or quaternion norm")
    lo, hi = stats["proprio_min"][5:9], stats["proprio_max"][5:9]
    nq = 2 * (result - lo) / (hi - lo) - 1
    audit = {"actual_quaternion": q.tolist(), "modeled_quaternion": result.tolist(),
             "actual_quaternion_sha256": array_sha(q), "modeled_quaternion_sha256": array_sha(result),
             "quaternion_dtype": q.dtype.str, "actual_norm": float(np.linalg.norm(q)),
             "modeled_norm": float(np.linalg.norm(result)), "norm_exact": bool(same_norm),
             "rotation_matrix_exact": bool(same_rotation), "rotation_matrix_max_abs_difference": 0.0,
             "protected_array_hashes": {k: before[k] for k in protected}, "protected_arrays_bitwise_equal": protected,
             "normalized_quaternion": nq.tolist(), "normalized_quaternion_outside_training_range": bool((np.abs(nq) > 1).any()),
             "input_changed": before != after}
    return modeled, audit


class ObservationAPI:
    def __init__(self, condition, stats, stream, api=inference):
        self.condition, self.stats, self.stream, self.api = condition, stats, stream, api
        self.rows = []

    def __getattr__(self, name):
        return getattr(self.api, name)

    def _call(self, method, model, vae, sampler, rand, obs, text, stats, **kwargs):
        modeled, audit = transform_observation(obs, self.condition, self.stats)
        audit.update(query_index=len(self.rows) + 1, native_method=method, condition=self.condition,
                     kwargs=kwargs, goal_embedding_sha256=__import__("hashlib").sha256(
                         text.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        self.rows.append(audit)
        self.stream.write(json.dumps(audit, allow_nan=False) + "\n")
        self.stream.flush()
        return getattr(self.api, method)(model, vae, sampler, rand, modeled, text, stats, **kwargs)

    def query(self, model, vae, sampler, rand, obs, text, stats, **kwargs):
        return self._call("query", model, vae, sampler, rand, obs, text, stats, **kwargs)

    def best_of_n(self, model, vae, sampler, rand, obs, text, stats, **kwargs):
        return self._call("best_of_n", model, vae, sampler, rand, obs, text, stats, **kwargs)


def verify_references(config):
    if config["planned_episodes"] != [{"best_of_n": n, "condition": c} for n, c in PLAN]:
        raise ValueError("Protocol must contain exactly the registered eight episodes")
    expected = {"suite": "libero_goal", "task_id": 0, "init_id": 0, "seed": 31,
                "environment_seed": 0, "max_steps": 300, "settle_steps": 10, "goal": GOAL,
                "cuda_memory_fraction": 0.28, "candidate_seeds": [31, 32]}
    if config["fixed_protocol"] != expected:
        raise ValueError("Registered native protocol changed")
    for path, sha in config["source_sha256"].items():
        inference.verify_artifact(ROOT / path, sha)
    refs = {}
    for n in (1, 2):
        entry = config["references"][str(n)]
        for path, sha in entry["artifact_sha256"].items():
            inference.verify_artifact(ROOT / path, sha)
        reference = json.loads((ROOT / entry["episode"]).read_text())
        for key in ("suite", "task_id", "init_id", "seed", "environment_seed", "max_steps", "settle_steps", "goal"):
            if type(reference.get(key)) is not type(expected[key]) or reference[key] != expected[key]:
                raise ValueError(f"Reference protocol mismatch: {key}")
        if reference["status"] != "completed" or reference["success"] is not True or reference["best_of_n"] != n:
            raise ValueError("Reference must be a complete successful native episode")
        provenance = reference["provenance"]
        if provenance["runner_sha256"] != base.digest(base.__file__) or provenance["helper_sha256"] != base.digest(inference.__file__):
            raise ValueError("Reference native implementation changed")
        for path, sha in provenance["source_sha256"].items():
            inference.verify_artifact(ROOT / path, sha)
        refs[n] = reference
    return refs


def enrich_trace(result, queries, output):
    """Preserve native trace; attach last queried quaternion to each actual action."""
    rows = [json.loads(s) for s in Path(result["trace"]).read_text().splitlines() if s]
    before_q = queries[0]["actual_quaternion"]
    with output.open("w") as handle:
        for row in rows:
            query = queries[row["chunk"] - 1]
            enriched = dict(row, actual_quaternion_before=before_q,
                            actual_quaternion_after=row["eef_quaternion"],
                            model_query_index=row["chunk"],
                            model_query_actual_quaternion=query["actual_quaternion"],
                            model_query_modeled_quaternion=query["modeled_quaternion"],
                            model_observation_reused_for_action_chunk=True,
                            model_queried_before_this_step=row["step"] == (row["chunk"] - 1) * 16 + 1)
            handle.write(json.dumps(enriched, allow_nan=False) + "\n")
            before_q = row["eef_quaternion"]
    return {"path": str(output), "sha256": base.digest(output), "steps": len(rows)}


def self_test(config, references):
    import io
    stats = inference.load_stats()
    with np.load(ROOT / "runs/cosmos_counterfactual_snapshots_v1/prefix_000/observation.npz") as archive:
        obs = {k: archive[k].copy() for k in archive.files if k != "sim_state"}
    frozen = copy.deepcopy(obs)
    checks = {}
    text = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)
    for condition in CONDITIONS:
        transformed, audit = transform_observation(obs, condition, stats)
        checks[f"{condition}_protected_and_orientation"] = all(audit["protected_arrays_bitwise_equal"].values()) and audit["norm_exact"] and audit["rotation_matrix_exact"]
        calls = []
        class FakeAPI:
            def query(self, *args, **kwargs):
                calls.append(("query", args, kwargs)); return "policy_result"
            def best_of_n(self, *args, **kwargs):
                calls.append(("best_of_n", args, kwargs)); return "planner_result"
        proxy = ObservationAPI(condition, stats, io.StringIO(), FakeAPI())
        a = proxy.query(None, None, None, None, obs, text, stats, num_steps=5, return_latent=True, image_transform="vertical", seed=31)
        b = proxy.best_of_n(None, None, None, None, obs, text, stats, n=2, seed=31)
        checks[f"{condition}_forwarded_native_calls"] = (a, b) == ("policy_result", "planner_result") and calls[0][2] == {"num_steps": 5, "return_latent": True, "image_transform": "vertical", "seed": 31} and calls[1][2] == {"n": 2, "seed": 31}
        checks[f"{condition}_only_observation_changed"] = all(call[1][5] is text and call[1][6] is stats and np.array_equal(call[1][4]["robot0_eef_quat"], transformed["robot0_eef_quat"]) for call in calls)
    checks["original_observation_unchanged"] = all(array_sha(obs[k]) == array_sha(v) for k, v in frozen.items())
    for q in (np.array([0., -1., 0., 0.]), np.array([0., 0., -1., 0.]), np.array([0., 0., 0., -1.]), np.array([1., 0., 0., 0.])):
        checks[f"lexicographic_tie_{q.tolist()}"] = np.array_equal(canonical(q), canonical(-q)) and np.array_equal(rotation_matrix(q), rotation_matrix(canonical(q)))
    for n, ref in references.items():
        checks[f"N{n}_reference_self_comparison"] = compare_episode(ref, ref)["all_exact"]
        altered = dict(ref, seed=32)
        checks[f"N{n}_seed_tamper_rejected"] = not compare_episode(ref, altered)["all_exact"]
    checks["eight_episodes_only"] = len(PLAN) == 8
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "passed", "checks": checks, "gpu_used": False, "egl_used": False,
            "scope": "Observation transformation/native-call forwarding and trace-comparator contracts; real clean parity is checked during the run",
            "runner_sha256": base.digest(__file__), "protocol_sha256": base.digest(CONFIG)}


def decision(results):
    rows = {(r["best_of_n"], r["condition"]): r for r in results}
    passed = (len(rows) == 8 and rows[2, "clean"]["success"] and
              rows[2, "sign_flip_then_x_canonical"]["success"] and
              not rows[2, "sign_flip"]["success"] and rows[1, "sign_flip"]["success"])
    return {"positive_gate_passed": bool(passed), "decision": "Go_to_fixed_candidate_path_attribution_only" if passed else "NoGo_stop_this_single_init_construction",
            "future_mediation_proven": False, "novelty_established": False,
            "independent_validation_completed": False, "independent_initial_states": 1,
            "clean_canonical_success_cost": {str(n): int(rows[n, "clean"]["success"]) - int(rows[n, "clean_x_canonical"]["success"]) for n in (1, 2)},
            "clean_canonical_steps_delta": {str(n): rows[n, "clean_x_canonical"]["environment_steps"] - rows[n, "clean"]["environment_steps"] for n in (1, 2)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    refs = verify_references(config)
    cpu = self_test(config, refs)
    OUT.mkdir(exist_ok=True, parents=True)
    test_path = OUT / "cpu_selftest.json"
    if test_path.exists() and json.loads(test_path.read_text()) != cpu:
        raise RuntimeError("Existing CPU gate uses different code/protocol")
    base.atomic_json(test_path, cpu)
    print(json.dumps(cpu), flush=True)
    if not args.run:
        return
    if (OUT / "summary.json").exists():
        raise FileExistsError("Do not resume/retry or overwrite this fixed eight-episode gate")
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") == "1":
        raise RuntimeError("This matched screen requires the recorded torch compatibility backend")
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    if task.language != GOAL:
        raise ValueError("Task goal changed")
    for path, sha in ((inference.MODEL, inference.MODEL_SHA256), (inference.TOKENIZER, inference.TOKENIZER_SHA256),
                      (base.CKPT_DIR / "libero_t5_embeddings.pkl", inference.TEXT_SHA256),
                      (base.CKPT_DIR / "libero_dataset_statistics.json", refs[2]["provenance"]["stats_sha256"])):
        inference.verify_artifact(path, sha)
    with (base.CKPT_DIR / "libero_t5_embeddings.pkl").open("rb") as handle:
        texts = pickle.load(handle)
    stats = inference.load_stats()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.28)
    started = time.monotonic()
    summary = {"status": "loading", "protocol": config, "protocol_sha256": base.digest(CONFIG), "results": [],
               "training_updates": 0, "api_requests": 0}
    path = OUT / "summary.json"
    base.atomic_json(path, summary)
    try:
        model, vae, sampler, rand = inference.load_model(device="cuda")
        from cosmos_policy_torch_compat import compatibility_metadata
        runtime = compatibility_metadata()
        if any(runtime != ref["provenance"]["runtime_compatibility"] for ref in refs.values()):
            raise RuntimeError("Actual runtime backend differs from the native reference")
        for n, condition in PLAN:
            directory = OUT / f"n{n}_{condition}"
            directory.mkdir(exist_ok=False)
            episode_args = SimpleNamespace(suite="libero_goal", seed=31, env_seed=0, max_steps=300,
                                           settle_steps=10, best_of_n=n, video_stride=0, out_dir=directory)
            provenance = copy.deepcopy(refs[n]["provenance"])
            provenance.update(runner_sha256=base.digest(__file__), native_runner_sha256=base.digest(base.__file__),
                              protocol_sha256=base.digest(CONFIG), runtime_compatibility=runtime,
                              observation_condition=condition, execution_offload=False,
                              cuda_memory_fraction=0.28, external_api_requests=0)
            query_path = directory / "model_observations.jsonl"
            with query_path.open("w") as stream:
                proxy = ObservationAPI(condition, stats, stream)
                original_api = base.inference
                original_writer = base.atomic_json
                def condition_writer(destination, payload):
                    # The frozen runner names all artifacts *_clean. Keep its
                    # trace protocol, but make running/error labels truthful.
                    payload = dict(payload)
                    payload["condition"] = condition
                    payload["legacy_filename_suffix"] = "_clean retained from frozen native runner; condition field is authoritative"
                    return original_writer(destination, payload)
                try:
                    base.inference = proxy
                    base.atomic_json = condition_writer
                    result = base.run_episode(model, vae, sampler, rand, task, 0, 0, texts, stats, episode_args, provenance)
                finally:
                    base.inference = original_api
                    base.atomic_json = original_writer
            result["condition"] = condition
            result["legacy_filename_suffix"] = "_clean retained from frozen native runner; condition field is authoritative"
            result["model_observations"] = str(query_path)
            result["model_observations_sha256"] = base.digest(query_path)
            result["enriched_trace"] = enrich_trace(result, proxy.rows, directory / "physical_and_model_trace.jsonl")
            result["model_observation_queries"] = len(proxy.rows)
            result["queries_outside_training_quaternion_range"] = sum(r["normalized_quaternion_outside_training_range"] for r in proxy.rows)
            if condition == "clean":
                result["full_native_reference_parity"] = compare_episode(refs[n], result)
                if not result["full_native_reference_parity"]["all_exact"]:
                    raise RuntimeError(f"N{n} clean differs from the complete native reference; abort")
            episode_path = directory / "libero_goal_task00_init00_clean.json"
            base.atomic_json(episode_path, result)
            summary["results"].append(result)
            summary.update(status="running", elapsed_seconds=time.monotonic() - started)
            base.atomic_json(path, summary)
            print(json.dumps({"group_result": {k: result[k] for k in ("best_of_n", "condition", "success", "environment_steps", "elapsed_seconds")},
                              "completed_episodes": len(summary["results"])}), flush=True)
            torch.cuda.empty_cache()
        summary.update(status="completed", gate=decision(summary["results"]))
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error),
                       gate={"decision": "technical_abort_not_task_failure", "do_not_expand": True})
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        base.atomic_json(path, summary)
    print(json.dumps({"summary": str(path), "gate": summary["gate"]}), flush=True)


if __name__ == "__main__":
    main()
