#!/usr/bin/env python3
"""Fixed-candidate Cosmos native future/value ranking, without outcome leakage.

This is an INTERNAL world-call context intervention, not an established external
attack interface. Two original-goal proposals are generated once (seeds 31/32).
Only the n_cond=5 forecast text changes; the n_cond=8 value text stays original.
NoLang is a zero-embedding ablation, NOT Bus-Cut. Real utility and forecast error
remain unknown until independent environment replay and an explicit join.

Use --self-test for CPU synthetic contract tests; no weights or simulator load.
Use --snapshot path/to/snapshot.json --preflight for input/resource validation.
Execution requires a successful official checkpoint Clean episode prefix from
cosmos_counterfactual_replay.capture_source_prefix(...), and a fresh --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from cosmos_counterfactual_replay import OBS_KEYS, array_sha, file_sha

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_GOAL = "open the middle drawer of the cabinet"
SWAP_GOAL = "put the bowl on the stove"
SEEDS = (31, 32)
CONDITIONS = ("clean", "goal_swap", "no_lang_zero_embedding")
LATENT_SHAPE = (1, 16, 9, 28, 28)
MODEL_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
STATS_SHA256 = "5b119a98ad7824507ddff3c6c7507ca244261ee416dcfa623876702160c580d3"
LIMITS = [
    "Internal split-call intervention; external attacker reachability is unproven.",
    "Goal-conditioned partial-observation forecasts may legitimately change hidden-state beliefs.",
    "Native predicted value is a selection score, never actual environment utility.",
    "Forecast divergence is not forecast error, attack success, or decision damage.",
    "Zero embedding is an untrained no-language representation ablation, not Bus-Cut.",
    "Bus-Cut, matched Control-Cut, and causal decision recovery are not implemented.",
    "Torch checkpoint adapter is used; full upstream Transformer Engine parity is not established.",
    "Only two candidates and a preselected successful trajectory prefix; no population claim.",
]


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def artifact_path(value, parent):
    value = Path(value)
    choices = [value] if value.is_absolute() else [parent / value, ROOT / value]
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(value)


def load_snapshot(path):
    """Validate input integrity; only the five current observations reach DiT/VAE."""
    path = Path(path).resolve()
    snapshot = json.loads(path.read_text())
    if snapshot.get("status") != "captured_original_native_prefix" or snapshot.get("task_success_at_prefix") is not False:
        raise ValueError("Expected a captured, nonterminal original native prefix")
    protocol = snapshot["protocol"]
    expected = {"suite": "libero_goal", "task_id": 0, "init_id": 0,
                "original_goal": ORIGINAL_GOAL, "seed": 31, "environment_seed": 0,
                "settle_steps": 10, "max_steps": 300}
    if any(type(protocol.get(key)) is not type(value) or protocol[key] != value
           for key, value in expected.items()):
        raise ValueError("Snapshot differs from preregistered task0/init0/seed31 protocol")
    prefix = snapshot["prefix_steps"]
    if type(prefix) is not int or prefix < 0 or prefix % 16 or prefix + 16 > 300:
        raise ValueError("Prefix must be a nonnegative native chunk boundary with 16 steps remaining")
    episode_path = artifact_path(snapshot["episode_json"], path.parent)
    if file_sha(episode_path) != snapshot["episode_sha256"]:
        raise ValueError("Source episode SHA256 changed")
    episode = json.loads(episode_path.read_text())
    if episode.get("status") != "completed" or episode.get("condition") != "clean" or episode.get("success") is not True:
        raise ValueError("Source must be a completed successful Clean episode")
    if episode.get("provenance", {}).get("checkpoint_sha256") != MODEL_SHA256:
        raise ValueError("Source is not the independent official NVIDIA checkpoint")
    for key, value in expected.items():
        source_key = "goal" if key == "original_goal" else key
        if type(episode.get(source_key)) is not type(value) or episode[source_key] != value:
            raise ValueError(f"Source protocol mismatch: {source_key}")
    if type(episode.get("environment_steps")) is not int or prefix >= episode["environment_steps"]:
        raise ValueError("Prefix is not within the successful source episode")
    state_sha = snapshot["state_sha256"]
    if not isinstance(state_sha, str) or len(state_sha) != 64 or any(c not in "0123456789abcdef" for c in state_sha):
        raise ValueError("Malformed physical state hash")
    npz_path = artifact_path(snapshot["observation_npz"], path.parent)
    if file_sha(npz_path) != snapshot["observation_npz_sha256"]:
        raise ValueError("Source observation file SHA256 changed")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(OBS_KEYS) | {"sim_state"}:
            raise ValueError("Unexpected raw-observation archive fields")
        arrays = {key: archive[key].copy() for key in archive.files}
    for key in OBS_KEYS[:2]:
        if arrays[key].shape != (256, 256, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f"Expected raw uint8 256px image: {key}")
    for key, length in zip(OBS_KEYS[2:], (2, 3, 4)):
        if arrays[key].shape != (length,) or arrays[key].dtype.kind != "f" or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Malformed proprio: {key}")
    if arrays["sim_state"].ndim != 1 or arrays["sim_state"].dtype.kind != "f" or not np.isfinite(arrays["sim_state"]).all():
        raise ValueError("Malformed simulator state")
    hashes = {key: array_sha(value) for key, value in arrays.items()}
    if snapshot.get("observation_array_hashes") != hashes:
        raise ValueError("Snapshot per-array hashes differ")
    contract = {"snapshot": str(path), "snapshot_sha256": file_sha(path),
                "episode_json": str(episode_path), "episode_sha256": file_sha(episode_path),
                "observation_npz": str(npz_path), "observation_npz_sha256": file_sha(npz_path),
                "state_sha256": state_sha, "array_sha256": hashes,
                "prefix_steps": prefix, "protocol": protocol,
                "source_checkpoint_provenance": episode["provenance"],
                "sim_state_used_as_model_input": False}
    contract["input_contract_sha256"] = json_sha(contract)
    return {key: arrays[key] for key in OBS_KEYS}, contract


def embeddings():
    import cosmos_policy_native_inference as native
    path = native.MODEL.parent / "libero_t5_embeddings.pkl"
    native.verify_artifact(path, native.TEXT_SHA256)
    with path.open("rb") as handle:
        values = pickle.load(handle)
    canonical, swapped = (torch.as_tensor(values[goal], dtype=torch.bfloat16).reshape(1, 512, 1024)
                           for goal in (ORIGINAL_GOAL, SWAP_GOAL))
    if not torch.isfinite(canonical).all() or not torch.isfinite(swapped).all() or torch.equal(canonical, swapped):
        raise ValueError("Invalid original/swap embeddings")
    return {"clean": canonical, "goal_swap": swapped,
            "no_lang_zero_embedding": torch.zeros_like(canonical)}


class AuditedRNG:
    def __init__(self, base, expected_seed):
        self.base, self.expected_seed, self.records = base, expected_seed, []

    def __call__(self, shape, seed=1):
        if seed != self.expected_seed or tuple(shape) != LATENT_SHAPE:
            raise RuntimeError("Unexpected native RNG seed/shape")
        noise = self.base(shape, seed=seed)
        if noise.dtype != torch.float32 or not torch.isfinite(noise).all():
            raise RuntimeError("Unexpected native noise dtype/content")
        self.records.append({"seed": seed, "shape": list(shape),
                             "unit_noise_sha256": tensor_sha(noise),
                             "sigma80_noise_sha256": tensor_sha(noise * 80.0)})
        return noise

    def finish(self):
        if len(self.records) != 1:
            raise RuntimeError("Native query did not consume exactly one audited initial-noise draw")
        return self.records[0]


class NativeBackend:
    def __init__(self, device="cuda", offload=False):
        import cosmos_policy_native_inference as native
        torch.backends.cudnn.deterministic = True
        self.native = native
        self.model, self.vae, self.sampler, self.rand = native.load_model(device="cpu" if offload else device)
        self.metadata = {"device": device, "offload": offload, "native_helper_sha256": file_sha(native.__file__),
                         "cudnn_deterministic": True, "full_transformer_engine_parity_established": False,
                         "compatibility_shim_sha256": file_sha(ROOT / "scripts/cosmos_policy_torch_compat.py")}
        if offload:
            from cosmos_policy_cuda_offload import prepare
            self.model, self.vae, self.manager, details = prepare(self.model, self.vae)
            self.metadata["offload_verification"] = details
        self.stats = native.load_stats()

    def encode(self, obs):
        return self.native.encode_observation(self.vae, obs, self.stats)

    def proposal(self, latent, text, seed, rand):
        return self.native.get_action(self.model, self.sampler, rand, latent, text, seed=seed, num_steps=5)

    def controls(self, normalized):
        return self.native.normalize_action(normalized, self.stats)[0]

    def future(self, latent, text, seed, rand):
        return self.native.get_future_state_prediction(self.model, self.sampler, rand, latent, text, seed=seed, num_steps=1)

    def value(self, latent, text, seed, rand):
        return self.native.get_value_prediction(self.model, self.sampler, rand, latent, text, seed=seed, num_steps=1)

    def decode(self, future, original):
        return self.native.decode_future_images(self.vae, future, original)


def checked_latent(value):
    if tuple(value.shape) != LATENT_SHAPE or not torch.isfinite(value).all():
        raise RuntimeError("Malformed/nonfinite sampled latent")


def checked_controls(value):
    value = np.asarray(value)
    if value.shape != (16, 7) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise RuntimeError("Expected exactly 16 finite real controls of width 7")
    return value.copy()


def checked_query(backend, phase, source, text, seed, prefix):
    """Pass private copies, assert input immutability and conditioned slots."""
    value, embedding = source.clone(), text.clone()
    input_sha, goal_sha = tensor_sha(value), tensor_sha(embedding)
    rng = AuditedRNG(backend.rand, seed)
    result = getattr(backend, phase)(value, embedding, seed, rng)
    output = result[-1] if isinstance(result, tuple) else result
    checked_latent(output)
    if tensor_sha(value) != input_sha or tensor_sha(embedding) != goal_sha:
        raise RuntimeError(f"{phase} mutated its input")
    if not torch.equal(output[:, :, :prefix].float(), source[:, :, :prefix].float()):
        raise RuntimeError(f"{phase} changed fixed conditioning prefix")
    return result, {"input_latent_sha256": input_sha, "text_sha256": goal_sha,
                    "conditioning_prefix_sha256": tensor_sha(source[:, :, :prefix].float()),
                    "conditioning_prefix_slots": prefix, "rng": rng.finish()}


def save_tensor(directory, relative, value):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu(), path)
    return {"path": relative, "file_sha256": file_sha(path), "tensor_sha256": tensor_sha(value),
            "shape": list(value.shape), "dtype": str(value.dtype)}


def save_array(directory, relative, value):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {"path": relative, "file_sha256": file_sha(path), "array_sha256": array_sha(value),
            "shape": list(value.shape), "dtype": value.dtype.str}


def rank_scores(scores):
    if len(scores) != 2 or not all(np.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise RuntimeError("Native value must be two finite probabilities in [0,1]")
    order = sorted(range(2), key=lambda i: (-scores[i], i))
    margin = abs(scores[0] - scores[1])
    return {"rank": order, "selected_index": order[0], "scores": scores,
            "margin": margin, "exact_tie": margin == 0.0, "near_tie_le_1e_6": margin <= 1e-6,
            "tie_rule": "first candidate index, matching native max"}


@torch.inference_mode()
def run_window(backend, obs, texts, contract, out_dir, *, decode_futures=False):
    directory = Path(out_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing nonempty output directory")
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "experiment": "fixed_candidate_native_future_value_ranking",
               "source": contract, "backend": backend.metadata, "limitations": LIMITS,
               "candidate_seeds": list(SEEDS), "conditions_planned": list(CONDITIONS),
               "original_goal": ORIGINAL_GOAL, "world_swap_goal": SWAP_GOAL,
               "horizon": 16, "native_steps": {"proposal": 5, "future": 1, "value": 1},
               "candidates": [], "conditions": {}, "forecast_error": None,
               "selected_action_utility": None, "bus_cut_decision_recovery": None,
               "real_environment_outcomes_status": "pending_independent_replay_and_join",
               "unimplemented_conditions": ["goal_swap_plus_bus_cut", "matched_control_cut"],
               "training_updates": 0, "api_requests": 0,
               "script_sha256": file_sha(__file__)}
    path = directory / "ranking.json"
    started = time.monotonic()
    write_json(path, summary)
    try:
        if set(obs) != set(OBS_KEYS) or set(texts) != set(CONDITIONS):
            raise ValueError("Unexpected observation or condition keys")
        goal_hashes = {condition: tensor_sha(texts[condition]) for condition in CONDITIONS}
        obs_hashes = {key: array_sha(value) for key, value in obs.items()}
        for condition in CONDITIONS:
            value = texts[condition]
            if tuple(value.shape) != (1, 512, 1024) or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("Expected finite native bf16 T5 embedding")
        if torch.equal(texts["clean"], texts["goal_swap"]) or torch.count_nonzero(texts["no_lang_zero_embedding"]):
            raise ValueError("Invalid condition embeddings")
        summary["goal_routing"] = {"policy": goal_hashes["clean"], "value": goal_hashes["clean"],
                                   "future": goal_hashes, "no_lang_definition": "all-zero T5 embedding; synthetic ablation"}
        observed, original = backend.encode({key: value.copy() for key, value in obs.items()})
        checked_latent(observed)
        checked_latent(original)
        summary["observed_latent"] = save_tensor(directory, "inputs/observed_latent.pt", observed)
        summary["original_vae_latent"] = save_tensor(directory, "inputs/original_vae_latent.pt", original)
        np.savez_compressed(directory / "inputs/observations.npz", **obs)
        summary["model_input_observations"] = {"path": "inputs/observations.npz",
                                               "file_sha256": file_sha(directory / "inputs/observations.npz"),
                                               "array_sha256": obs_hashes}
        summary["embeddings"] = {condition: save_tensor(directory, f"inputs/{condition}_text.pt", texts[condition])
                                 for condition in CONDITIONS}
        action_latents, clean_futures = [], []
        # Generate ONCE before any intervention; never use future outcomes here.
        for index, seed in enumerate(SEEDS):
            (normalized, latent), audit = checked_query(backend, "proposal", observed, texts["clean"], seed, 4)
            if normalized.shape != (1, 16, 7) or not torch.isfinite(normalized).all():
                raise RuntimeError("Malformed normalized action proposal")
            controls = checked_controls(backend.controls(normalized))
            record = {"index": index, "seed": seed, "audit": audit,
                      "actions": save_array(directory, f"candidates/candidate_{index:02d}_actions.npy", controls),
                      "normalized_actions": save_tensor(directory, f"candidates/candidate_{index:02d}_normalized.pt", normalized),
                      "action_latent": save_tensor(directory, f"candidates/candidate_{index:02d}_latent.pt", latent)}
            summary["candidates"].append(record)
            action_latents.append(latent.clone())
            write_json(path, summary)
        summary["candidate_controls_distinct"] = (summary["candidates"][0]["actions"]["array_sha256"] !=
                                                   summary["candidates"][1]["actions"]["array_sha256"])
        for condition in CONDITIONS:
            rows, scores = [], []
            for index, seed in enumerate(SEEDS):
                candidate = summary["candidates"][index]
                future, future_audit = checked_query(backend, "future", action_latents[index], texts[condition], seed, 5)
                (value, value_latent), value_audit = checked_query(backend, "value", future, texts["clean"], seed, 8)
                if future_audit["rng"] != candidate["audit"]["rng"] or value_audit["rng"] != candidate["audit"]["rng"]:
                    raise RuntimeError("Native proposal/future/value did not reuse the candidate's fixed initial noise")
                if value.numel() != 1:
                    raise RuntimeError("Expected one native scalar value")
                score = float(value.item())
                scores.append(score)
                row = {"candidate_index": index, "seed": seed, "predicted_value": score,
                       "candidate_actions_sha256": candidate["actions"]["array_sha256"],
                       "future_audit": future_audit, "value_audit": value_audit,
                       "future_latent": save_tensor(directory, f"{condition}/candidate_{index:02d}_future.pt", future),
                       "value_latent": save_tensor(directory, f"{condition}/candidate_{index:02d}_value.pt", value_latent)}
                if condition == "clean":
                    clean_futures.append(future[:, :, 5:8].detach().cpu().float().clone())
                    row["forecast_divergence_from_clean_latent_mse"] = 0.0
                else:
                    reference = summary["conditions"]["clean"]["candidates"][index]
                    for phase in ("future_audit", "value_audit"):
                        if row[phase]["rng"] != reference[phase]["rng"]:
                            raise RuntimeError("Matched condition RNG changed")
                    row["forecast_divergence_from_clean_latent_mse"] = float(
                        (future[:, :, 5:8].detach().cpu().float() - clean_futures[index]).square().mean())
                if decode_futures:
                    row["decoded_future_images"] = {key: save_array(directory, f"{condition}/candidate_{index:02d}_{key}.npy", pixels)
                                                    for key, pixels in backend.decode(future, original).items()}
                    row["forecast_error_target_protocol"] = "Independent replay at exact t+16; compare decoded RGB to native preprocess_observation(actual future obs). No latent target shortcut."
                rows.append(row)
                print(json.dumps({"phase": "ranking", "condition": condition, "candidate": index,
                                  "predicted_value": score, "elapsed_seconds": time.monotonic() - started}), flush=True)
                del future, value_latent
            ranking = rank_scores(scores)
            ranking["candidates"] = rows
            ranking["selected_actions"] = summary["candidates"][ranking["selected_index"]]["actions"]
            reference = summary["conditions"].get("clean", ranking)
            ranking["ranking_changed_from_clean"] = ranking["rank"] != reference["rank"]
            ranking["selected_index_changed_from_clean"] = ranking["selected_index"] != reference["selected_index"]
            ranking["selected_action_utility"] = None
            summary["conditions"][condition] = ranking
            write_json(path, summary)
        if {key: array_sha(value) for key, value in obs.items()} != obs_hashes:
            raise RuntimeError("Source observations changed")
        if {key: tensor_sha(texts[key]) for key in CONDITIONS} != goal_hashes:
            raise RuntimeError("Cached goal embeddings changed")
        for index, latent in enumerate(action_latents):
            if tensor_sha(latent) != summary["candidates"][index]["action_latent"]["tensor_sha256"]:
                raise RuntimeError("Fixed candidate latent changed")
        summary["replay_interface"] = {"module": "scripts/cosmos_counterfactual_replay.py",
            "episode_json": contract.get("episode_json"), "prefix_steps": contract.get("prefix_steps"),
            "source_snapshot": contract.get("snapshot"), "repeat": 2,
            "candidate_actions": [record["actions"] for record in summary["candidates"]],
            "continuation_required_for_full_task_utility": True,
            "join_keys": ["source.state_sha256", "candidate index", "actions.array_sha256"],
            "scores_or_predictions_are_never_inputs_to_replay": True}
        summary["status"] = "predictions_completed_true_outcomes_pending"
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(path, summary)
    return path


def self_test():
    """Synthetic endpoint contracts only; NO learned-model inference or env."""
    import contextlib
    import io
    checks = []
    class Stub:
        metadata = {"synthetic_cpu_endpoint_stub": True}
        def __init__(self, drift=False):
            self.calls, self.drift = [], drift
        def rand(self, shape, seed=1):
            return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape).astype(np.float32))
        def encode(self, obs):
            self.calls.append(("encode", None, None))
            return torch.zeros(LATENT_SHAPE), torch.zeros(LATENT_SHAPE)
        def proposal(self, source, text, seed, rand):
            self.calls.append(("proposal", seed, float(text.flatten()[0])))
            rand(tuple(source.shape), seed=seed)
            result = source.clone()
            result[:, :, 4] = seed - 31
            return torch.full((1, 16, 7), (seed - 31) / 10), result
        def controls(self, normalized):
            return normalized.numpy()[0].astype(np.float64)
        def future(self, source, text, seed, rand):
            self.calls.append(("future", seed, float(text.flatten()[0])))
            rand(tuple(source.shape), seed=seed)
            result = source.clone()
            result[:, :, 5:8] = (seed - 31.5) * float(text.flatten()[0])
            if self.drift:
                result[:, :, 4] += 1
            return result
        def value(self, source, text, seed, rand):
            self.calls.append(("value", seed, float(text.flatten()[0])))
            rand(tuple(source.shape), seed=seed)
            result = source.clone()
            score = float(source[:, :, 5:8].mean()) / 2 + 0.5
            result[:, :, 8] = score * 2 - 1
            return torch.tensor([score]), result
    obs = {key: np.zeros((256, 256, 3), dtype=np.uint8) if index < 2 else np.zeros((2, 3, 4)[index - 2], dtype=np.float64)
           for index, key in enumerate(OBS_KEYS)}
    texts = {"clean": torch.ones(1, 512, 1024, dtype=torch.bfloat16),
             "goal_swap": -torch.ones(1, 512, 1024, dtype=torch.bfloat16),
             "no_lang_zero_embedding": torch.zeros(1, 512, 1024, dtype=torch.bfloat16)}
    initialized = torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory(prefix="cosmos_ranking_cpu_") as temporary, contextlib.redirect_stdout(io.StringIO()):
        directory = Path(temporary)
        backend = Stub()
        path = run_window(backend, obs, texts, {"synthetic_test": True}, directory / "ranking")
        result = json.loads(path.read_text())
        counts = {phase: sum(call[0] == phase for call in backend.calls) for phase in ("encode", "proposal", "future", "value")}
        assert counts == {"encode": 1, "proposal": 2, "future": 6, "value": 6}
        assert all(call[2] == 1 for call in backend.calls if call[0] in ("proposal", "value"))
        checks.append("one encoding, exactly two shared proposals, original policy/value goal in all six evaluations")
        assert [result["conditions"][condition]["selected_index"] for condition in CONDITIONS] == [1, 0, 0]
        assert result["forecast_error"] is None and result["selected_action_utility"] is None
        assert result["bus_cut_decision_recovery"] is None and len(result["unimplemented_conditions"]) == 2
        checks.append("synthetic ranking flips and native first-index tie handling; no invented true utility/error/cuts")
        for record in result["candidates"]:
            actions = np.load(path.parent / record["actions"]["path"])
            assert array_sha(actions) == record["actions"]["array_sha256"]
        checks.append("candidate action round-trip SHA256 agrees with independent replay utility")
        faulty = Stub(drift=True)
        try:
            run_window(faulty, obs, texts, {}, directory / "drift")
        except RuntimeError as error:
            assert "fixed conditioning" in str(error)
            assert not any(call[0] == "value" for call in faulty.calls)
        else:
            raise AssertionError("Failed to reject action conditioning drift")
        checks.append("forecast action-prefix drift rejected before value scoring")
        for malformed in (np.zeros((15, 7)), np.full((16, 7), np.nan)):
            try:
                checked_controls(malformed)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Malformed controls accepted")
        checks.append("malformed horizon and nonfinite controls rejected")
        protocol = {"suite": "libero_goal", "task_id": 0, "init_id": 0, "original_goal": ORIGINAL_GOAL,
                    "seed": 31, "environment_seed": 0, "settle_steps": 10, "max_steps": 300}
        episode = {**{key: value for key, value in protocol.items() if key != "original_goal"},
                   "goal": ORIGINAL_GOAL, "status": "completed", "condition": "clean", "success": True,
                   "environment_steps": 120, "provenance": {"checkpoint_sha256": MODEL_SHA256}}
        ep_path = directory / "episode.json"
        write_json(ep_path, episode)
        arrays = {**obs, "sim_state": np.zeros(5, dtype=np.float64)}
        npz_path = directory / "observation.npz"
        np.savez(npz_path, **arrays)
        snapshot = {"status": "captured_original_native_prefix", "task_success_at_prefix": False,
                    "protocol": protocol, "prefix_steps": 16, "episode_json": str(ep_path),
                    "episode_sha256": file_sha(ep_path), "observation_npz": str(npz_path),
                    "observation_npz_sha256": file_sha(npz_path), "state_sha256": "1" * 64,
                    "observation_array_hashes": {key: array_sha(value) for key, value in arrays.items()}}
        sp_path = directory / "snapshot.json"
        write_json(sp_path, snapshot)
        loaded, _ = load_snapshot(sp_path)
        assert set(loaded) == set(OBS_KEYS)
        episode["success"] = "true"
        write_json(ep_path, episode)
        for update_hash in (False, True):
            if update_hash:
                snapshot["episode_sha256"] = file_sha(ep_path)
                write_json(sp_path, snapshot)
            try:
                load_snapshot(sp_path)
            except ValueError:
                pass
            else:
                raise AssertionError("Tampered/invalid success source accepted")
        checks.append("source SHA tampering and string-valued success rejected; simulator state excluded from model")
    assert torch.cuda.is_initialized() == initialized
    checks.append("no CUDA initialization, real model loading, environment rollout or API calls")
    return {"status": "passed", "kind": "synthetic_cpu_contract_test_not_experiment", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--decode-futures", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.snapshot is None or (not args.preflight and args.out_dir is None):
        parser.error("--snapshot required; execution also requires --out-dir")
    if args.offload and args.device != "cuda":
        parser.error("--offload computes on CUDA, so use --device cuda")
    obs, contract = load_snapshot(args.snapshot)
    texts = embeddings()
    import cosmos_policy_native_inference as native
    stats_path = native.MODEL.parent / "libero_dataset_statistics.json"
    native.verify_artifact(stats_path, STATS_SHA256)
    native.verify_artifact(native.MODEL, MODEL_SHA256)
    native.verify_artifact(native.TOKENIZER, native.TOKENIZER_SHA256)
    contract["inference_provenance"] = {"checkpoint_sha256": MODEL_SHA256,
        "checkpoint_revision": native.CHECKPOINT_REVISION, "source_revision": native.SOURCE_REVISION,
        "tokenizer_sha256": native.TOKENIZER_SHA256, "stats_sha256": STATS_SHA256,
        "text_cache_sha256": native.TEXT_SHA256, "helper_sha256": file_sha(native.__file__)}
    contract["inference_provenance"]["source_sha256"] = {
        relative: file_sha(ROOT / relative) for relative in (
            "scripts/cosmos_policy_native_inference.py", "scripts/cosmos_policy_codecs.py",
            "scripts/cosmos_policy_wan_vae.py", "scripts/cosmos_policy_torch_compat.py",
            "scripts/alphabrain_checkpoint_inference.py", "scripts/cosmos_policy_cuda_offload.py",
            "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/cosmos_sampler.py",
            "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/official_dit/minimal_v4_dit.py")}
    if args.preflight:
        print(json.dumps({"status": "cpu_preflight_passed_no_inference", "source": contract,
                          "candidate_seeds": SEEDS, "conditions": CONDITIONS, "limitations": LIMITS}, indent=2))
        return
    backend = NativeBackend(device=args.device, offload=args.offload)
    path = run_window(backend, obs, texts, contract, args.out_dir, decode_futures=args.decode_futures)
    print(json.dumps({"ranking": str(path), "true_outcomes": "pending independent replay"}))


if __name__ == "__main__":
    main()
