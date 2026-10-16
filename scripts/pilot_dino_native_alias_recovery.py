#!/usr/bin/env python3
"""One separately frozen native episode with trusted candidate binding."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

from pilot_dino_native_alias_pair import (
    ROOT, ResourceDeferred, bootstrap, canonical_boundaries, equal, gpu_info,
    rng_state, take_snapshot, write,
)

CONFIG = ROOT / "configs/dino_native_alias_recovery_20261015.json"


class InvalidExperiment(RuntimeError):
    pass


def check(ok, message):
    if not ok:
        raise InvalidExperiment(message)


def canonical_writer(untrusted, pre, boundaries, torch):
    """Writer sees only an independent request copy, never the trusted reference."""
    request = untrusted.reshape(*untrusted.shape[:2], 5, 4)
    raw = pre.denormalize_actions(request)
    for axis in range(3):
        for side in (-1, 1):
            mask = (raw[..., axis].abs() > 1) & (raw[..., axis].sign() == side)
            request[..., axis][mask] = boundaries[axis, side]
    return request.reshape_as(untrusted)


def reconstruct(cfg, source, out, torch, np, mujoco, activity):
    from pilot_dino_metaworld_clean_gate import physical, setup_episode
    env, initial, goal, setup = setup_episode(cfg, 4, out / "reconstructed_setup")
    activity.update(expert_setup_controls=100, root_warmup_controls=1)
    try:
        with np.load(source / "setup/setup.npz", allow_pickle=False) as archive:
            saved = {k: archive[k].copy() for k in archive.files}
        with np.load(out / "reconstructed_setup/setup.npz", allow_pickle=False) as archive:
            check(set(archive.files) == set(saved), "Reconstructed setup schema differs")
            check(all(archive[k].dtype == saved[k].dtype and np.array_equal(archive[k], saved[k]) for k in saved), "Reconstructed setup arrays differ")
        check((source / "setup/expert_trace.jsonl").read_bytes() == (out / "reconstructed_setup/expert_trace.jsonl").read_bytes(), "Reconstructed expert trace differs")
        raw = env.proprio_env.unwrapped
        with np.load(source / "snapshot.npz", allow_pickle=False) as archive:
            snapshot = {k: archive[k].copy() for k in archive.files}
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        current = np.empty(mujoco.mj_stateSize(raw.model, spec), np.float64)
        mujoco.mj_getState(raw.model, raw.data, current, spec)
        check(np.array_equal(current, snapshot["integration"]), "Reconstructed full integration differs before restoration")
        check(np.array_equal(raw.model.site_pos, snapshot["site_pos"]), "Reconstructed site positions differ")
        check(all(np.array_equal(v, snapshot["physical_" + k]) for k, v in physical(env).items()), "Reconstructed stored physical fields differ")
        raw.model.site_pos[:] = snapshot["site_pos"]
        mujoco.mj_setState(raw.model, raw.data, snapshot["integration"], spec)
        mujoco.mj_forward(raw.model, raw.data)
        mujoco.mj_setState(raw.model, raw.data, snapshot["integration"], spec)
        torch.set_rng_state(torch.from_numpy(snapshot["torch_cpu_rng"]))
        torch.cuda.set_rng_state(torch.from_numpy(snapshot["torch_cuda_rng"]), 0)
        check(np.array_equal(env.proprio_env.render().transpose(2, 0, 1)[None], saved["initial_pixels"]), "Restored initial pixels differ")
        previous = raw._prev_obs.copy()
        actual_state = raw._get_obs().astype(np.float32)
        raw._prev_obs = previous
        # _get_obs constructs a new history-bearing39D observation; its previous
        # half must not replace the actual saved warmup observation. That entire
        # observation was checked by setup-array equality above.
        check(np.array_equal(actual_state[:4], saved["initial_proprio"][0]), "Restored physical proprio differs")
        check(np.array_equal(raw._target_pos, saved["target"]), "Original real task target differs")
        check(np.array_equal(initial["visual"].cpu().numpy(), saved["initial_pixels"]) and np.array_equal(initial["proprio"].cpu().numpy(), saved["initial_proprio"]), "Initial observation differs")
        check(np.array_equal(goal["visual"].cpu().numpy(), saved["goal_pixels"]) and np.array_equal(goal["proprio"].cpu().numpy(), saved["goal_proprio"]), "Original goal observation differs")
        # Fresh setup reconstructed raw fields/history and Python/NumPy RNG;
        # saved integration/site/Torch RNG were explicitly reinstalled above.
        restore, metadata = take_snapshot(env, initial, torch, np, mujoco, out)
        metadata["source"] = "Same seed4 native setup reconstruction plus saved complete integration/site/Torch RNG; not a portable full Python snapshot"
        metadata["source_setup_all_arrays_and_expert_trace_exact"] = True
        metadata["source_integration_and_physical_fields_exact_before_restore"] = True
        restore()
        return env, initial.clone(), goal.clone(), setup, metadata
    except BaseException:
        env.close()
        raise


def main(config_path, override_out):
    protocol = json.loads(config_path.read_text())
    out = (override_out or ROOT / protocol["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    source = ROOT / protocol["source_run"]
    started, env = time.monotonic(), None
    report = {"status": "resource_precheck", "started_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": str(config_path),
              "source_run": str(source), "purpose": protocol["purpose"], "queries": [], "outcome": None, "decision": None, "checkpoint_loads": {},
              "activity": {"policy_controls": 0, "native_unrolls": 0, "writer_calls": 0, "binding_calls": 0, "network_attempts": 0, "training_updates": 0}}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        write(out / "summary.json", report)
    def expired(signum, frame):
        raise ResourceDeferred("Frozen1200-second recovery wall limit reached")
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    persist()
    try:
        previous = json.loads((source / "summary.json").read_text())
        check(previous["status"] == "completed_native_alias_pair" and previous["decision"]["consider_separate_recovery_pilot"], "Source did not complete the registered recovery trigger")
        references = {arm: previous["arms"][arm]["outcome"] for arm in ("clean_identity_copy", "canonical_request")}
        report["reference_outcomes"] = references
        del previous  # No source query/actions are used by the recovery policy.
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["memory.free"] < protocol["runtime"]["minimum_start_free_mib"]:
            raise ResourceDeferred("Startup actual free VRAM below5632MiB; no model/environment")
        import numpy as np
        import torch
        import mujoco
        from tensordict import TensorDict
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, trace_step, outcome, MODEL, DINO_WEIGHTS
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        check(mujoco.__version__ == "3.3.0", "Pinned MuJoCo missing")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        total = torch.cuda.get_device_properties(0).total_memory
        cap = int(protocol["runtime"]["allocator_cap_gib"] * 1024**3)
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        report["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__, "mujoco": mujoco.__version__,
                             "allocator_cap_bytes": cap, "allocator_fraction": cap / total, "allocator_excludes_context": True,
                             "threads": 4, "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": True}
        original_load = torch.load
        def load_once(path, *args, **kwargs):
            key = str(Path(path).resolve())
            check(key in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and key not in report["checkpoint_loads"], "Unexpected/repeated checkpoint")
            report["checkpoint_loads"][key] = 1
            return original_load(path, *args, **kwargs)
        def no_network(*args, **kwargs):
            report["activity"]["network_attempts"] += 1
            raise InvalidExperiment("No network authorized")
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            guards.enter_context(patch.object(torch, "load", side_effect=load_once))
            cfg = cfg_for(out, "cuda")
            report["status"] = "loading_local_checkpoint"; persist()
            model, pre, report["strict_load"] = load_model(cfg, "cuda")
            check(len(report["checkpoint_loads"]) == 2, "Expected exactly two checkpoints")
            boundaries = canonical_boundaries(pre, torch)
            report["canonical_boundaries"] = {f"axis{a}_sign{s}": float(v) for (a, s), v in boundaries.items()}
            report["status"] = "reconstructing_same_seed4_setup"; persist()
            env, obs, goal, report["setup"], report["snapshot"] = reconstruct(cfg, source, out, torch, np, mujoco, report["activity"])
            directory = out / "recovery"; directory.mkdir()
            agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
            agent.set_goal(goal.clone())
            check((agent.planner.num_samples, agent.planner.iterations, agent.planner.num_elites, agent.planner.horizon, agent.planner.num_act_stepped) == (300, 15, 10, 6, 3) and agent.planner.max_norms is None, "Native planner budget changed")
            goal_cpu = {k: v.detach().cpu().clone() for k, v in agent.goal_state_enc.items()}
            with np.load(source / "clean_identity_copy/goal_encoding.npz", allow_pickle=False) as archive:
                check(all(np.array_equal(v.numpy(), archive[k]) for k, v in goal_cpu.items()), "Encoded original goal differs")
            np.savez_compressed(directory / "goal_encoding.npz", **{k: v.numpy() for k, v in goal_cpu.items()})
            with np.load(source / "clean_identity_copy/first_proposal.npz", allow_pickle=False) as archive:
                first_reference = {k: archive[k].copy() for k in ("original", "context_visual", "context_proprio")}
            native_unroll, active = model.unroll, {}
            def guarded_unroll(z, act_suffix=None, **kwargs):
                check(not kwargs and act_suffix is not None and act_suffix.dtype == torch.float32 and act_suffix.shape[1] in (1, 300) and act_suffix.shape[2] == 20, "Unexpected native request")
                before_rng = rng_state(torch, np)
                local_rng = agent.local_gpu_generator.get_state().clone()
                trusted = act_suffix.detach().cpu().clone()
                trusted_saved = trusted.clone()
                untrusted = trusted.clone()
                check(untrusted.untyped_storage().data_ptr() != trusted.untyped_storage().data_ptr(), "Writer received shared trusted storage")
                rewritten = canonical_writer(untrusted, pre, boundaries, torch)
                report["activity"]["writer_calls"] += 1
                raw_original = pre.denormalize_actions(trusted.reshape(*trusted.shape[:2], 5, 4))
                request = rewritten.reshape_as(raw_original)
                raw_written = pre.denormalize_actions(request)
                mask = torch.zeros_like(raw_original, dtype=torch.bool); mask[..., :3] = raw_original[..., :3].abs() > 1
                check(torch.isfinite(raw_original).all() and torch.isfinite(raw_written).all(), "Nonfinite request")
                check(torch.equal(request[~mask], trusted.reshape_as(request)[~mask]), "Writer changed untouched normalized coordinates")
                check(torch.equal(raw_written[..., :3].clamp(-1, 1), raw_original[..., :3].clamp(-1, 1)) and torch.equal(raw_written[..., 3], raw_original[..., 3]), "Canonical writer changed effective physical controls")
                check(torch.equal(trusted, trusted_saved), "Writer modified private trusted reference")
                # The only reference for repair is this current native candidate.
                # Neither archived Clean actions nor model output is consulted.
                bound_request = trusted.clone().to(act_suffix.device)
                report["activity"]["binding_calls"] += 1
                check(torch.equal(bound_request.detach().cpu(), trusted_saved) and bound_request.untyped_storage().data_ptr() != act_suffix.untyped_storage().data_ptr(), "Trusted binding failed")
                check(equal(before_rng, rng_state(torch, np), np) and torch.equal(local_rng, agent.local_gpu_generator.get_state()), "Writer/binding consumed RNG")
                context = {k: v.detach().cpu().clone() for k, v in z.items()}
                if "context" not in active:
                    active["context"] = context
                else:
                    check(all(torch.equal(v, active["context"][k]) for k, v in context.items()), "Context changed within query")
                query = active["query"]; stats = query["request_stats"]
                if query["index"] == 1 and stats["unroll_calls"] == 0:
                    check(np.array_equal(trusted.numpy(), first_reference["original"]), "First proposal differs from source Clean")
                    check(all(np.array_equal(v.numpy(), first_reference["context_" + k]) for k, v in context.items()), "First context differs from source Clean")
                    report["initial_goal_context_and_first_proposal_exact"] = True
                    np.savez_compressed(directory / "first_proposal.npz", trusted=trusted.numpy(), writer=rewritten.numpy(), bound=bound_request.detach().cpu().numpy(), **{"context_" + k: v.numpy() for k, v in context.items()})
                stats["unroll_calls"] += 1
                stats["candidate_calls" if trusted.shape[1] == 300 else "mean_calls"] += 1
                stats["saturated_xyz_coordinates"] += int(mask.sum())
                stats["writer_changed_coordinates"] += int((trusted != rewritten).sum())
                stats["binding_restored_coordinates"] += int((trusted != rewritten).sum())
                report["activity"]["native_unrolls"] += 1
                prediction = native_unroll(z, act_suffix=bound_request)
                check(torch.equal(act_suffix.detach().cpu(), trusted_saved), "Original planner candidates mutated")
                check(all(torch.equal(v.detach().cpu(), context[k]) for k, v in z.items()), "Latent context mutated")
                check(all(torch.equal(v.detach().cpu(), goal_cpu[k]) for k, v in agent.goal_state_enc.items()), "Original goal mutated")
                for value in prediction.values():
                    check(all(bool(torch.isfinite(frame).all()) for frame in value), "Nonfinite native forecast")
                if trusted.shape[1] == 1:
                    active["last_mean_trusted"] = trusted_saved
                    active["last_mean_writer"] = rewritten.clone()
                    active["last_mean_bound"] = bound_request.detach().cpu().clone()
                return prediction
            agent.planner.unroll = guarded_unroll
            rows = []
            report["status"] = "running_single_recovery"; persist()
            with (directory / "trace.jsonl").open("x") as handle:
                while env.steps_left() > 0:
                    qi = len(report["queries"]) + 1
                    steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                    query = {"index": qi, "status": "planning", "elapsed_before": env.elapsed_steps(), "horizon": min(6, steps_left), "executed_controls": 0,
                             "request_stats": {"unroll_calls": 0, "candidate_calls": 0, "mean_calls": 0, "saturated_xyz_coordinates": 0, "writer_changed_coordinates": 0, "binding_restored_coordinates": 0}}
                    report["queries"].append(query); active.clear(); active["query"] = query; persist()
                    torch.cuda.synchronize(); tick = time.monotonic()
                    normalized = agent.act(obs, steps_left=steps_left)
                    torch.cuda.synchronize(); query["planning_seconds"] = time.monotonic() - tick
                    check(query["request_stats"]["candidate_calls"] == query["request_stats"]["mean_calls"] == 15, "Native call budget changed")
                    generator_state = agent.local_gpu_generator.get_state().clone()
                    with np.load(source / f"clean_identity_copy/query{qi}_forecast_cache.npz", allow_pickle=False) as archive:
                        check(np.array_equal(generator_state.numpy(), archive["local_cuda_generator_after"]), "Local CEM RNG differs from source Clean")
                    query["source_local_generator_exact"] = True
                    mean = agent.planner._prev_mean.detach().cpu()
                    check(torch.equal(normalized, mean[:3]) and torch.equal(mean.unsqueeze(1), active["last_mean_trusted"]), "Action was not current native final mean")
                    actions = pre.denormalize_actions(normalized.reshape(-1, 4))
                    check(torch.isfinite(actions).all(), "Nonfinite executable controls")
                    query.update(status="executing", normalized_macro_actions=normalized.tolist(), physical_actions=actions.tolist(), minimum_cost_per_iteration=agent._prev_losses.cpu().tolist())
                    forecast = agent._predicted_best_encs_over_iterations[-1]
                    cache = {"context_" + k: v.numpy() for k, v in active["context"].items()}
                    cache.update({"mean_trusted": active["last_mean_trusted"].numpy(), "mean_writer": active["last_mean_writer"].numpy(), "mean_bound": active["last_mean_bound"].numpy(), "local_cuda_generator_after": generator_state.numpy()})
                    cache.update({"mean_forecast_" + k: v.detach().cpu().numpy() for k, v in forecast.items()})
                    actual_pixels, actual_proprio, offsets = [], [], []
                    for index, action in enumerate(actions):
                        visual, info, row = trace_step(env, action, qi, index)
                        check(row["physical"]["target"] == report["setup"]["arrays"]["target"], "True task target changed")
                        rows.append(row); handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                        report["activity"]["policy_controls"] += 1; query["executed_controls"] += 1
                        obs = TensorDict({"visual": visual, "proprio": info["proprio"]})
                        if (index + 1) % 5 == 0:
                            actual_pixels.append(visual.cpu().numpy().copy()); actual_proprio.append(info["proprio"].cpu().numpy().copy()); offsets.append(index + 1)
                        if row["done"]:
                            break
                    cache.update(actual_pixels=np.stack(actual_pixels), actual_proprio=np.stack(actual_proprio), actual_control_offsets=np.asarray(offsets, np.int64))
                    np.savez_compressed(directory / f"query{qi}_forecast_cache.npz", **cache)
                    query.update(status="completed", actual_forecast_aligned_control_offsets=offsets, final_reward=rows[-1]["reward"], final_success=rows[-1]["success"], final_reach_distance=rows[-1]["reach_distance"])
                    write(directory / "queries.json", report["queries"]); persist()
                    print(json.dumps({"query": qi, "planning_seconds": query["planning_seconds"], "environment_step": env.elapsed_steps(), "success": rows[-1]["success"], "reach_distance": rows[-1]["reach_distance"]}), flush=True)
                    if qi == 1:
                        report["first_query_episode_cost_estimate_seconds"] = query["planning_seconds"] * 37 / 6
                        if report["first_query_episode_cost_estimate_seconds"] > 1100:
                            raise ResourceDeferred("First actual recovery query implies planning above1100s; prefix preserved")
                    if rows[-1]["done"]:
                        break
            check([q["elapsed_before"] for q in report["queries"]] == protocol["fixed"]["expected_query_elapsed"] and [q["horizon"] for q in report["queries"]] == protocol["fixed"]["expected_horizons"], "Native seven-query schedule changed")
            check(report["activity"]["native_unrolls"] == report["activity"]["writer_calls"] == report["activity"]["binding_calls"] == 210, "Expected one forecast per writer/binding call")
            report["outcome"] = outcome(rows, directory)
            report["outcome"].update(terminal_native_reward=rows[-1]["reward"], first_success_policy_control=next((r["policy_step"] for r in rows if r["success"]), None), success_controls=sum(r["success"] for r in rows),
                                     reward_sum_including_shared_warmup=sum(r["reward"] for r in rows) + report["setup"]["warmup_reward_excluded_from_policy_return"])
            recovered = report["outcome"]["reward_sum"]
            clean = references["clean_identity_copy"]["reward_sum"]; canonical = references["canonical_request"]["reward_sum"]
            report["decision"] = {"utility_recovered_to_clean_within_1e_minus6_tolerance": recovered >= clean - 1e-6,
                                  "recovery_minus_clean_return": recovered - clean, "recovery_minus_canonical_return": recovered - canonical,
                                  "recovered_fraction_unclipped": (recovered - canonical) / (clean - canonical),
                                  "interpretation": "Conventional trusted-current-candidate integrity restoration in one exploratory episode; no source action substitution, extra inference, novelty or population-safety claim."}
            report["status"] = "completed_single_native_alias_recovery"
    except ResourceDeferred as error:
        report.update(status="resource_deferred", reason=str(error))
    except BaseException as error:
        oom = type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()
        report.update(status="resource_deferred" if oom else "invalid_reconstruction_or_runtime", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if not oom:
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            try:
                env.close()
            except Exception as error:
                report["environment_close_error"] = str(error)
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["activity"]["total_explicit_environment_controls"] = sum(report["activity"].get(k, 0) for k in ("expert_setup_controls", "root_warmup_controls", "policy_controls"))
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "decision": report["decision"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bootstrap()
    main(args.config.resolve(), args.out)
