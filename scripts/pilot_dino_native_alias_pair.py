#!/usr/bin/env python3
"""One frozen native DINO CEM pair with copied canonical forecast requests."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dino_native_alias_pair_20261015.json"
ARMS = ("clean_identity_copy", "canonical_request")


class ResourceDeferred(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def bootstrap():
    if os.environ.get("DINO_NATIVE_ALIAS_BOOTSTRAPPED") != "1":
        os.environ.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                          LIBGL_ALWAYS_SOFTWARE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                          DINO_NATIVE_ALIAS_BOOTSTRAPPED="1")
        os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["PYTHONPATH"] = str(ROOT / ".deps/dino_mujoco330") + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "Frozen GPU/allocator setting changed")
    require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "OSMesa required")
    sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
    sys.path.insert(0, str(ROOT / ".deps/dino_mujoco330"))


def gpu_info():
    fields = ("memory.total", "memory.free", "memory.used", "utilization.gpu")
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    return dict(zip(fields, (int(v.strip()) for v in result.stdout.strip().split(","))))


def equal(a, b, np):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (tuple, list, deque)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y, np) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k], np) for k in a)
    if isinstance(a, np.random.Generator):
        return isinstance(b, np.random.Generator) and equal(a.bit_generator.state, b.bit_generator.state, np)
    if isinstance(a, np.random.RandomState):
        return isinstance(b, np.random.RandomState) and equal(a.get_state(), b.get_state(), np)
    if isinstance(a, random.Random):
        return isinstance(b, random.Random) and a.getstate() == b.getstate()
    return type(a) is type(b) and a == b


def rng_state(torch, np):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().copy(),
            torch.cuda.get_rng_state(0).cpu().numpy().copy())


def take_snapshot(env, initial, torch, np, mujoco, out):
    from pilot_dino_metaworld_clean_gate import physical
    raw = env.proprio_env.unwrapped
    require(raw.reward_function_version == "v2" and env.elapsed_steps() == 1, "Expected audited reach-v2 root after one warmup")
    def simple(v, depth=0):
        if isinstance(v, (type(None), bool, int, float, str, np.generic, np.ndarray, np.random.Generator, np.random.RandomState, random.Random)):
            return True
        if depth >= 3:
            return False
        if isinstance(v, (list, tuple, deque)):
            return all(simple(x, depth + 1) for x in v)
        if isinstance(v, dict):
            return all(isinstance(k, (str, int)) and simple(x, depth + 1) for k, x in v.items())
        return False
    nodes, node = [], env
    while True:
        fields = {k: copy.deepcopy(v) for k, v in node.__dict__.items() if simple(v)}
        nodes.append((node, fields))
        if node is raw:
            break
        require("env" in node.__dict__, "Unexpected wrapper topology")
        node = node.__dict__["env"]
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(raw.model, spec), np.float64)
    mujoco.mj_getState(raw.model, raw.data, integration, spec)
    sites, baseline = raw.model.site_pos.copy(), physical(env)
    global_rng = rng_state(torch, np)
    np.savez_compressed(out / "snapshot.npz", integration=integration, site_pos=sites,
                        torch_cpu_rng=global_rng[2], torch_cuda_rng=global_rng[3],
                        **{"physical_" + k: v for k, v in baseline.items()})
    metadata = {"source": "Actual seed4 native setup, no historical-state reconstruction", "integration_spec": "mjSTATE_INTEGRATION",
                "integration_length": len(integration), "node_fields": [{"class": type(n).__name__, "fields": sorted(f)} for n, f in nodes],
                "global_rng": ["Python", "NumPy", "Torch CPU", "Torch CUDA0"], "restores": 0}
    def restore():
        for node, fields in nodes:
            for key in list(node.__dict__):
                if key not in fields and simple(node.__dict__[key]):
                    del node.__dict__[key]
            for key, value in fields.items():
                setattr(node, key, copy.deepcopy(value))
        raw.model.site_pos[:] = sites
        mujoco.mj_setState(raw.model, raw.data, integration, spec)
        mujoco.mj_forward(raw.model, raw.data)
        mujoco.mj_setState(raw.model, raw.data, integration, spec)
        restored = np.empty_like(integration)
        mujoco.mj_getState(raw.model, raw.data, restored, spec)
        require(np.array_equal(restored, integration), "Integration snapshot restoration differs")
        for node, fields in nodes:
            require(all(equal(node.__dict__[k], v, np) for k, v in fields.items()), "Raw/wrapper history or RNG restoration differs")
        require(all(np.array_equal(physical(env)[k], v) for k, v in baseline.items()), "Restored physical fields differ")
        require(np.array_equal(env.proprio_env.render().transpose(2, 0, 1)[None], initial["visual"].cpu().numpy()), "Restored initial pixels differ")
        random.setstate(global_rng[0]); np.random.set_state(global_rng[1])
        torch.set_rng_state(torch.from_numpy(global_rng[2].copy()))
        torch.cuda.set_rng_state(torch.from_numpy(global_rng[3].copy()), 0)
        require(equal(rng_state(torch, np), global_rng, np), "Global RNG restore differs")
        metadata["restores"] += 1
        write(out / "snapshot_metadata.json", metadata)
    return restore, metadata


def canonical_boundaries(pre, torch):
    boundaries = {}
    for axis in range(3):
        for side in (-1, 1):
            raw = torch.zeros(4, dtype=torch.float32); raw[axis] = side
            value = pre.normalize_actions(raw)[axis]
            outward = torch.tensor(float("inf") * side, dtype=torch.float32)
            def valid(z):
                probe = torch.zeros(4, dtype=torch.float32); probe[axis] = z
                return float(pre.denormalize_actions(probe)[axis].clamp(-1, 1)) == side
            count = 0
            while not valid(value):
                require(count < 16, "Outward boundary limit")
                value = torch.nextafter(value, outward); count += 1
            count = 0
            while valid(torch.nextafter(value, -outward)):
                require(count < 16, "Inward boundary limit")
                value = torch.nextafter(value, -outward); count += 1
            boundaries[axis, side] = value.clone()
    return boundaries


def main(config_path, override_out):
    protocol = json.loads(config_path.read_text())
    out = (override_out or ROOT / protocol["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started, env = time.monotonic(), None
    summary = {"status": "resource_precheck", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": str(config_path), "hypothesis": protocol["hypothesis"], "disclosure": protocol["disclosure"],
               "arms": {}, "noop": None, "decision": None, "checkpoint_loads": {},
               "activity": {"unroll_calls": 0, "policy_controls": 0, "noop_controls": 0, "network_attempts": 0, "training_updates": 0}}
    def persist():
        summary["elapsed_seconds"] = time.monotonic() - started
        write(out / "summary.json", summary)
    def expired(signum, frame):
        raise ResourceDeferred("Frozen2700-second total wall limit reached")
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    persist()
    try:
        summary["gpu_at_start"] = gpu_info(); persist()
        if summary["gpu_at_start"]["memory.free"] < protocol["runtime"]["minimum_start_free_mib"]:
            raise ResourceDeferred(f"Startup free VRAM below{protocol['runtime']['minimum_start_free_mib']}MiB; no model or environment created")
        import numpy as np
        import torch
        import mujoco
        from tensordict import TensorDict
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, setup_episode, trace_step, outcome, MODEL, DINO_WEIGHTS
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        require(mujoco.__version__ == "3.3.0", "Pinned MuJoCo overlay missing")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        cap = int(protocol["runtime"]["allocator_cap_gib"] * 1024**3)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        summary["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__, "torch_path": torch.__file__, "mujoco": mujoco.__version__,
                              "allocator_cap_bytes": cap, "allocator_fraction": cap / total, "allocator_is_not_total_process_cap": True,
                              "total_gpu_bytes": total, "threads": 4, "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": True}
        cfg = cfg_for(out, "cuda")
        original_load = torch.load
        def once(path, *args, **kwargs):
            key = str(Path(path).resolve())
            require(key in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and key not in summary["checkpoint_loads"], "Unexpected/repeated checkpoint load")
            summary["checkpoint_loads"][key] = 1
            return original_load(path, *args, **kwargs)
        def no_network(*args, **kwargs):
            summary["activity"]["network_attempts"] += 1
            raise RuntimeError("No network authorized")
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            guards.enter_context(patch.object(torch, "load", side_effect=once))
            summary["status"] = "loading_local_checkpoint"; persist()
            model, pre, summary["strict_load"] = load_model(cfg, "cuda")
            require(len(summary["checkpoint_loads"]) == 2, "Expected two existing checkpoints")
            boundaries = canonical_boundaries(pre, torch)
            summary["canonical_boundary_values"] = {f"axis{a}_sign{s}": float(v) for (a, s), v in boundaries.items()}
            summary["status"] = "native_seed4_setup"; persist()
            env, initial, goal, summary["setup"] = setup_episode(cfg, 4, out / "setup")
            initial, goal = initial.clone(), goal.clone()
            restore, summary["snapshot"] = take_snapshot(env, initial, torch, np, mujoco, out)
            summary["activity"]["expert_setup_controls"] = 100
            summary["activity"]["root_warmup_controls"] = 1
            noop_dir = out / "noop"; noop_dir.mkdir()
            summary["status"] = "same_root_noop"; persist()
            restore(); rows = []
            with (noop_dir / "trace.jsonl").open("x") as handle:
                while env.steps_left() > 0:
                    _, _, row = trace_step(env, torch.zeros(4), 0, 0)
                    require(row["physical"]["target"] == summary["setup"]["arrays"]["target"], "True task target changed during no-op")
                    rows.append(row); summary["activity"]["noop_controls"] += 1
                    handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                    if row["done"]:
                        break
            summary["noop"] = outcome(rows, noop_dir)
            summary["noop"]["terminal_native_reward"] = rows[-1]["reward"]
            persist()
            pair_reference = {}
            native_unroll = model.unroll
            for arm in ARMS:
                restore()
                directory = out / arm; directory.mkdir()
                record = {"status": "running", "queries": [], "outcome": None}
                summary["arms"][arm] = record
                agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
                agent.set_goal(goal.clone())
                require(agent.planner.num_samples == 300 and agent.planner.iterations == 15 and agent.planner.num_elites == 10 and agent.planner.max_norms is None, "Native planner changed")
                goal_cpu = {k: v.detach().cpu().clone() for k, v in agent.goal_state_enc.items()}
                if arm == ARMS[0]:
                    pair_reference["goal"] = goal_cpu
                    pair_reference["generator_start"] = agent.local_gpu_generator.get_state().clone()
                else:
                    require(all(torch.equal(v, pair_reference["goal"][k]) for k, v in goal_cpu.items()), "Paired encoded goal differs")
                    require(torch.equal(agent.local_gpu_generator.get_state(), pair_reference["generator_start"]), "Paired initial CEM RNG differs")
                np.savez_compressed(directory / "goal_encoding.npz", **{k: v.numpy() for k, v in goal_cpu.items()})
                active = {}
                def copied_unroll(z, act_suffix=None, **kwargs):
                    require(not kwargs and act_suffix is not None, "Unexpected native unroll signature")
                    require(act_suffix.dtype == torch.float32 and act_suffix.shape[1] in (1, 300) and act_suffix.shape[2] == 20, "Native request schema changed")
                    before_rng = rng_state(torch, np)
                    local_rng = agent.local_gpu_generator.get_state().clone()
                    original = act_suffix.detach().cpu().clone()
                    request = original.reshape(*original.shape[:2], 5, 4).clone()
                    raw = pre.denormalize_actions(request)
                    require(torch.isfinite(raw).all(), "Nonfinite original request")
                    mask = torch.zeros_like(request, dtype=torch.bool); mask[..., :3] = raw[..., :3].abs() > 1
                    if arm == ARMS[1]:
                        for axis in range(3):
                            for side in (-1, 1):
                                chosen = mask[..., axis] & (raw[..., axis].sign() == side)
                                request[..., axis][chosen] = boundaries[axis, side]
                    canonical_raw = pre.denormalize_actions(request)
                    require(torch.equal(request[~mask], original.reshape_as(request)[~mask]), "Untouched normalized values changed")
                    require(torch.equal(canonical_raw[..., :3].clamp(-1, 1), raw[..., :3].clamp(-1, 1)), "Effective clipped XYZ differs")
                    require(torch.equal(canonical_raw[..., 3], raw[..., 3]), "Raw gripper differs")
                    require(equal(before_rng, rng_state(torch, np), np) and torch.equal(local_rng, agent.local_gpu_generator.get_state()), "Rewrite consumed RNG")
                    context_cpu = {k: v.detach().cpu().clone() for k, v in z.items()}
                    if "context" not in active:
                        active["context"] = context_cpu
                    else:
                        require(all(torch.equal(v, active["context"][k]) for k, v in context_cpu.items()), "Query context changed")
                    if active["query"]["index"] == 1 and active["query"]["request_stats"]["unroll_calls"] == 0:
                        if arm == ARMS[0]:
                            pair_reference["first_proposal"] = original.clone()
                            pair_reference["first_context"] = context_cpu
                        else:
                            require(torch.equal(original, pair_reference["first_proposal"]), "Paired first proposal differs")
                            require(all(torch.equal(v, pair_reference["first_context"][k]) for k, v in context_cpu.items()), "Paired first context differs")
                        np.savez_compressed(directory / "first_proposal.npz", original=original.numpy(), rewritten=request.reshape_as(original).numpy(), **{"context_" + k: v.numpy() for k, v in context_cpu.items()})
                    stats = active["query"]["request_stats"]
                    stats["unroll_calls"] += 1
                    stats["candidate_calls" if original.shape[1] == 300 else "mean_calls"] += 1
                    stats["saturated_xyz_coordinates"] += int(mask.sum())
                    stats["changed_normalized_coordinates"] += int((request != original.reshape_as(request)).sum())
                    stats["maximum_original_raw_abs_xyz"] = max(stats["maximum_original_raw_abs_xyz"], float(raw[..., :3].abs().max()))
                    stats["maximum_rewritten_raw_abs_xyz"] = max(stats["maximum_rewritten_raw_abs_xyz"], float(canonical_raw[..., :3].abs().max()))
                    summary["activity"]["unroll_calls"] += 1
                    prediction = native_unroll(z, act_suffix=request.reshape_as(original).to(act_suffix.device))
                    require(torch.equal(act_suffix.detach().cpu(), original), "Caller action tensor mutated")
                    require(all(torch.equal(v.detach().cpu(), context_cpu[k]) for k, v in z.items()), "Caller latent context mutated")
                    require(all(torch.equal(v.detach().cpu(), goal_cpu[k]) for k, v in agent.goal_state_enc.items()), "Encoded goal mutated")
                    for value in prediction.values():
                        require(all(bool(torch.isfinite(frame).all()) for frame in value), "Nonfinite predicted features")
                    if original.shape[1] == 1:
                        active["last_mean_original"] = original
                        active["last_mean_request"] = request.reshape_as(original).clone()
                    return prediction
                agent.planner.unroll = copied_unroll
                obs, rows = initial.clone(), []
                summary["status"] = "running_" + arm; persist()
                with (directory / "trace.jsonl").open("x") as handle:
                    while env.steps_left() > 0:
                        qi = len(record["queries"]) + 1
                        steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                        query = {"index": qi, "elapsed_before": env.elapsed_steps(), "horizon": min(6, steps_left), "status": "planning",
                                 "request_stats": {"unroll_calls": 0, "candidate_calls": 0, "mean_calls": 0, "saturated_xyz_coordinates": 0, "changed_normalized_coordinates": 0,
                                                   "maximum_original_raw_abs_xyz": 0., "maximum_rewritten_raw_abs_xyz": 0., "effective_actions_and_immutable_inputs_exact": True}, "executed_controls": 0}
                        record["queries"].append(query); active.clear(); active["query"] = query; persist()
                        torch.cuda.synchronize(); tick = time.monotonic()
                        normalized = agent.act(obs, steps_left=steps_left)
                        torch.cuda.synchronize(); query["planning_seconds"] = time.monotonic() - tick
                        require(query["request_stats"]["candidate_calls"] == query["request_stats"]["mean_calls"] == 15, "Native per-query call budget changed")
                        state = agent.local_gpu_generator.get_state().clone()
                        if arm == ARMS[0]:
                            pair_reference["generator_query" + str(qi)] = state
                            query["paired_local_generator_exact"] = None
                            query["local_generator_comparison_status"] = "reference_saved"
                        else:
                            require(torch.equal(state, pair_reference["generator_query" + str(qi)]), "Paired CEM random stream differs")
                            query["paired_local_generator_exact"] = True
                            query["local_generator_comparison_status"] = "exact_to_clean_reference"
                        mean = agent.planner._prev_mean.detach().cpu()
                        require(torch.equal(normalized, mean[:3]) and torch.equal(mean.unsqueeze(1), active["last_mean_original"]), "Executed action is not native final mean")
                        actions = pre.denormalize_actions(normalized.reshape(-1, 4))
                        require(torch.isfinite(actions).all(), "Nonfinite executable controls")
                        query.update(status="executing", normalized_macro_actions=normalized.tolist(), physical_actions=actions.tolist(), minimum_cost_per_iteration=agent._prev_losses.cpu().tolist())
                        forecast = agent._predicted_best_encs_over_iterations[-1]
                        cache = {"context_" + k: v.numpy() for k, v in active["context"].items()}
                        cache.update({"mean_original": active["last_mean_original"].numpy(), "mean_request": active["last_mean_request"].numpy(), "local_cuda_generator_after": state.numpy()})
                        cache.update({"mean_forecast_" + k: v.detach().cpu().numpy() for k, v in forecast.items()})
                        real_pixels, real_proprio, real_steps = [], [], []
                        for index, action in enumerate(actions):
                            visual, info, row = trace_step(env, action, qi, index)
                            require(row["physical"]["target"] == summary["setup"]["arrays"]["target"], "True task target changed during policy execution")
                            rows.append(row); handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                            query["executed_controls"] += 1; summary["activity"]["policy_controls"] += 1
                            obs = TensorDict({"visual": visual, "proprio": info["proprio"]})
                            if (index + 1) % 5 == 0:
                                real_pixels.append(visual.cpu().numpy().copy()); real_proprio.append(info["proprio"].cpu().numpy().copy()); real_steps.append(index + 1)
                            if row["done"]:
                                break
                        cache.update(actual_pixels=np.stack(real_pixels), actual_proprio=np.stack(real_proprio), actual_control_offsets=np.asarray(real_steps, np.int64))
                        np.savez_compressed(directory / f"query{qi}_forecast_cache.npz", **cache)
                        query.update(status="completed", actual_forecast_aligned_control_offsets=real_steps, final_reward=rows[-1]["reward"], final_success=rows[-1]["success"], final_reach_distance=rows[-1]["reach_distance"])
                        write(directory / "queries.json", record["queries"]); persist()
                        print(json.dumps({"arm": arm, "query": qi, "planning_seconds": query["planning_seconds"], "environment_step": env.elapsed_steps(), "success": rows[-1]["success"], "reach_distance": rows[-1]["reach_distance"]}), flush=True)
                        if arm == ARMS[0] and qi == 1:
                            summary["first_query_pair_cost_estimate_seconds"] = query["planning_seconds"] * (37 / 6) * 2
                            if summary["first_query_pair_cost_estimate_seconds"] > 2400:
                                raise ResourceDeferred("First actual Clean query implies pair planning cost above2400s; executed prefix preserved")
                        if rows[-1]["done"]:
                            break
                require([q["elapsed_before"] for q in record["queries"]] == protocol["native_planner"]["expected_elapsed_before"], "Native replan schedule changed")
                require([q["horizon"] for q in record["queries"]] == protocol["native_planner"]["expected_horizons"], "Native horizon schedule changed")
                record["outcome"] = outcome(rows, directory)
                record["outcome"].update(terminal_native_reward=rows[-1]["reward"], reward_sum_including_shared_warmup=sum(r["reward"] for r in rows) + summary["setup"]["warmup_reward_excluded_from_policy_return"])
                record["status"] = "completed"; persist()
                agent.planner.unroll = native_unroll
                del agent
            clean, canonical = (summary["arms"][a]["outcome"] for a in ARMS)
            difference = canonical["reward_sum"] - clean["reward_sum"]
            success_loss = clean["final_success"] and not canonical["final_success"]
            relative_drop = -difference / clean["reward_sum"] if clean["reward_sum"] > 0 else None
            meaningful_drop = clean["reward_sum"] > summary["noop"]["reward_sum"] and relative_drop is not None and relative_drop >= .10
            summary["decision"] = {"canonical_minus_clean_return": difference, "relative_return_drop": relative_drop, "final_success_loss": success_loss,
                                   "clean_return_above_noop": clean["reward_sum"] > summary["noop"]["reward_sum"],
                                   "consider_separate_recovery_pilot": bool(success_loss or meaningful_drop),
                                   "stop_without_expansion": bool(not success_loss and difference >= 0),
                                   "interpretation": "One exploratory native closed-loop pair. No population significance/ASR, no oracle gate, no forecast-error comparison across different executed trajectories."}
            summary["status"] = "completed_native_alias_pair"
    except ResourceDeferred as error:
        summary.update(status="resource_deferred", reason=str(error))
    except BaseException as error:
        oom = type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()
        summary.update(status="resource_deferred" if oom else "failed_closed_native_alias_pair", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if not oom:
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            try:
                env.close()
            except Exception as error:
                summary["environment_close_error"] = str(error)
        if "torch" in locals() and torch.cuda.is_initialized():
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summary["activity"]["total_explicit_environment_controls"] = sum(summary["activity"].get(k, 0) for k in ("expert_setup_controls", "root_warmup_controls", "noop_controls", "policy_controls"))
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"], "decision": summary["decision"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bootstrap()
    main(args.config.resolve(), args.out)
