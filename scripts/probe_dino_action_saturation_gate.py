#!/usr/bin/env python3
"""One CPU-only, frozen-bank DINO action-saturation gate; never run native CEM."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import islice
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/dino_action_saturation_gate_20261015.json"
SIM_FIELDS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")
ARMS = ("original", "inflated_saturated_xyz", "canonical_saturated_xyz")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bootstrap():
    overlay = ROOT / ".deps/dino_mujoco330"
    if os.environ.get("DINO_ALIAS_BOOTSTRAPPED") != "1":
        os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                          LIBGL_ALWAYS_SOFTWARE="1", DINO_ALIAS_BOOTSTRAPPED="1")
        os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["PYTHONPATH"] = str(overlay) + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be disabled")
    require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "OSMesa required")
    sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
    sys.path.insert(0, str(overlay))


def make_bank(protocol, pre, torch, np):
    bank, ideals, traces = [], [], []
    for source in protocol["candidate_bank"]["native_sources"]:
        directory = ROOT / source
        queries = json.loads((directory / "queries.json").read_text())
        with (directory / "trace.jsonl").open() as handle:
            rows = [json.loads(line) for line in islice(handle, 30)]
        require(len(rows) == 30 and [q["elapsed_before"] for q in queries[:2]] == [1, 16], "Unexpected native prefix")
        norm = torch.tensor([a for q in queries[:2] for a in q["normalized_macro_actions"]], dtype=torch.float32).reshape(30, 4)
        raw = torch.tensor([r["action"] for r in rows], dtype=torch.float32)
        require(norm.shape == (30, 4) and torch.equal(pre.denormalize_actions(norm), raw), "Native cached normalization/trace mismatch")
        bank.append(norm); ideals.append(raw); traces.append(rows)
    zero = torch.zeros_like(ideals[0])
    reverse = ideals[0].clone(); reverse[:, :3].neg_()
    zero_norm = pre.normalize_actions(zero)
    reverse_norm = pre.normalize_actions(reverse)
    reverse_norm[:, 3] = bank[0][:, 3]
    bank.extend((zero_norm, reverse_norm)); ideals.extend((zero, reverse))
    original = torch.stack(bank)
    raw = pre.denormalize_actions(original)
    require(torch.isfinite(original).all() and torch.isfinite(raw).all(), "Nonfinite fixed bank")
    sat = torch.zeros_like(original, dtype=torch.bool)
    sat[:, :, :3] = raw[:, :, :3].abs() > 1

    # Boundaries are found using the actual native float32 denormalizer.
    boundaries = {}
    for axis in range(3):
        for side in (-1, 1):
            value = torch.zeros(4, dtype=torch.float32); value[axis] = side
            z = pre.normalize_actions(value)[axis]
            outward = torch.tensor(float("inf") * side, dtype=torch.float32)
            inward = -outward
            def valid(x):
                probe = torch.zeros(4, dtype=torch.float32); probe[axis] = x
                return float(pre.denormalize_actions(probe)[axis].clamp(-1, 1)) == side
            count = 0
            while not valid(z):
                require(count < 16, "Canonical outward boundary search exceeded frozen limit")
                z = torch.nextafter(z, outward); count += 1
            inward_count = 0
            while valid(torch.nextafter(z, inward)):
                require(inward_count < 16, "Canonical inward boundary search exceeded frozen limit")
                z = torch.nextafter(z, inward); inward_count += 1
            boundaries[(axis, side)] = z.clone()

    def canonical(norm):
        result = norm.clone()
        request = pre.denormalize_actions(norm)
        mask = torch.zeros_like(sat); mask[:, :, :3] = request[:, :, :3].abs() > 1
        require(torch.equal(mask, sat), "Canonical intervention mask changed")
        for axis in range(3):
            for side in (-1, 1):
                use = sat[:, :, axis] & (torch.sign(request[:, :, axis]) == side)
                result[:, :, axis][use] = boundaries[(axis, side)]
        return result

    inflated = original.clone()
    desired = raw.clone()
    desired[sat] = raw[sat].sign() * (raw[sat].abs() + 2)
    inflated[sat] = pre.normalize_actions(desired)[sat]
    canonical_original = canonical(original)
    canonical_inflated = canonical(inflated)
    require(torch.equal(canonical_original, canonical_inflated), "Clean/Inflated canonical controls differ")
    norms = torch.stack((original, inflated, canonical_original))
    requests = pre.denormalize_actions(norms)
    for i, name in enumerate(ARMS):
        require(torch.equal(norms[i][~sat], original[~sat]), name + ": untouched normalized coordinates changed")
        require(torch.equal(requests[i, :, :, 3], raw[:, :, 3]), name + ": gripper changed")
        require(torch.equal(requests[i, :, :, :3].clamp(-1, 1), raw[:, :, :3].clamp(-1, 1)), name + ": effective XYZ changed")
    require(not bool(sat[3].any()), "Zero-roundtrip negative control unexpectedly saturated")
    info = {
        "candidate_names": protocol["candidate_bank"]["order"],
        "native_trace_denormalization_exact": [True] * 3,
        "constructed_max_error_vs_ideal_raw": [float((raw[i] - ideals[i]).abs().max()) for i in (3, 4)],
        "saturated_xyz_coordinates_per_candidate": sat.sum(dim=(1, 2)).tolist(),
        "boundary_normalized_values": {f"axis{a}_sign{s}": float(z) for (a, s), z in boundaries.items()},
        "unchanged_normalized_coordinates_exact": True,
        "effective_xyz_and_gripper_exact": True,
        "clean_and_inflated_canonical_inputs_exact": True,
        "scope": "Fixed new open-loop bank; all requests frozen before environment/model outputs"
    }
    return norms, requests, traces[0], info


def reconstruct_and_snapshot(env, cached, mujoco, torch, np):
    from pilot_dino_metaworld_clean_gate import physical
    raw = env.proprio_env.unwrapped
    require(raw.reward_function_version == "v2", "Only the audited native reach-v2 reward is supported")
    for name in SIM_FIELDS:
        getattr(raw.data, name)[:] = cached["sim_" + name]
    raw.data.time = float(cached["sim_time"])
    raw.model.site_pos[:] = cached["model_site_pos"]
    raw._target_pos = cached["target"].copy()
    raw._last_rand_vec = cached["rand_vec"].copy()
    raw._prev_obs = cached["previous_observation"].copy()
    raw._last_stable_obs = cached["initial_state"].astype(np.float64).copy()
    raw._did_see_sim_exception = False
    raw.curr_path_length = int(cached["elapsed"])
    env.set_elapsed_steps(int(cached["elapsed"]))
    require(env.elapsed_steps() == 1, "Root must be after exactly one historical warmup")
    raw.maxReachDist = float(np.linalg.norm(raw.init_tcp - raw._target_pos))
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qacc_warmstart[:] = cached["sim_qacc_warmstart"]
    env._frames.clear(); env._frames.append(cached["initial_pixels"][0].copy())
    env._proprios.clear(); env._proprios.append(cached["initial_proprio"][0].copy())
    pixels = env.proprio_env.render().transpose(2, 0, 1)[None].copy()
    require(np.array_equal(pixels, cached["initial_pixels"]), "Initial rendered pixels differ from cached root")
    before = raw._prev_obs.copy(); obs = raw._get_obs().astype(np.float32); raw._prev_obs = before
    require(np.array_equal(obs[:4], cached["initial_proprio"][0]), "Initial proprio differs from cached root")
    for name in SIM_FIELDS:
        require(np.array_equal(getattr(raw.data, name), cached["sim_" + name]), "Initial integrator field differs: " + name)

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(raw.model, spec), dtype=np.float64)
    mujoco.mj_getState(raw.model, raw.data, integration, spec)
    # Retain every simple raw-environment field, including reward parameters and RNG.
    allowed = (type(None), bool, int, float, str, np.ndarray, np.random.Generator, np.random.RandomState)
    raw_fields = {k: copy.deepcopy(v) for k, v in raw.__dict__.items() if isinstance(v, allowed)}
    wrappers = []
    node = env
    while node is not raw:
        fields = {k: copy.deepcopy(node.__dict__[k]) for k in ("_elapsed_steps", "_max_episode_steps", "_frames", "_proprios") if k in node.__dict__}
        wrappers.append((node, fields))
        require("env" in node.__dict__, "Unexpected wrapper topology")
        node = node.__dict__["env"]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    sites = raw.model.site_pos.copy()
    baseline_physical = physical(env)

    def restore():
        for k, v in raw_fields.items():
            setattr(raw, k, copy.deepcopy(v))
        for wrapper, fields in wrappers:
            for k, v in fields.items():
                setattr(wrapper, k, copy.deepcopy(v))
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        raw.model.site_pos[:] = sites
        mujoco.mj_setState(raw.model, raw.data, integration, spec)
        mujoco.mj_forward(raw.model, raw.data)
        mujoco.mj_setState(raw.model, raw.data, integration, spec)
        check = np.empty_like(integration)
        mujoco.mj_getState(raw.model, raw.data, check, spec)
        require(np.array_equal(check, integration), "Complete integration snapshot restoration failed")
        now = physical(env)
        require(all(np.array_equal(now[k], v) for k, v in baseline_physical.items()), "Root physical/wrapper snapshot changed")
        require(np.array_equal(env.proprio_env.render().transpose(2, 0, 1)[None], cached["initial_pixels"]), "Restored root pixels changed")

    metadata = {"source": "cached root reconstructed before in-process snapshot",
                "integration_state_spec": "mjSTATE_INTEGRATION", "integration_state_length": len(integration),
                "raw_python_fields": sorted(raw_fields),
                "wrapper_fields": [{"class": type(w).__name__, "fields": sorted(f)} for w, f in wrappers],
                "initial_pixels_and_proprio_exact": True, "historical_source_prefix_verified": False,
                "note": "MuJoCo derived fields recomputed by mj_forward; integration fields then restored exactly. Constructor may reset/simulate internally; no extra explicit reset/warmup is performed."}
    return restore, metadata


def replay_bank(env, restore, requests, historical, out, report, persist, torch, np):
    import mujoco
    from pilot_dino_metaworld_clean_gate import trace_step, physical
    gt_pixels, gt_proprio = [], []
    utilities = []
    raw = env.proprio_env.unwrapped
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(raw.model, state_spec)
    for candidate in range(5):
        base_rows, base_frames, base_phys, base_integration = [], [], [], []
        milestones_pixels, milestones_proprio = [], []
        for arm_index, arm in enumerate(ARMS):
            restore()
            directory = out / f"candidate{candidate}" / arm
            directory.mkdir(parents=True, exist_ok=False)
            rows = []
            with (directory / "trace.jsonl").open("x") as handle:
                for step in range(30):
                    visual, info, row = trace_step(env, requests[arm_index, candidate, step].clone(), 0, step)
                    report["activity"]["environment_steps"] += 1
                    pixels = visual.cpu().numpy().copy()
                    state = physical(env)
                    integration = np.empty(state_size, dtype=np.float64)
                    mujoco.mj_getState(raw.model, raw.data, integration, state_spec)
                    row["candidate"] = candidate; row["representation"] = arm
                    handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                    require(not env.proprio_env.unwrapped._did_see_sim_exception, "Native simulator reported an exception")
                    require(not row["done"] and not row["truncated"], "Unexpected termination inside the30-control bank")
                    if arm_index == 0:
                        if candidate == 0:
                            old = historical[step]
                            keys = ("action", "reward", "done", "truncated", "success", "reach_distance", "tcp_before", "tcp_after", "state", "physical", "pixels", "environment_step", "policy_step")
                            mismatch = [k for k in keys if row[k] != old[k]]
                            require(not mismatch, f"Historical source1 mismatch at control{step+1}: {mismatch}")
                        base_rows.append(row); base_frames.append(pixels); base_phys.append(state)
                        base_integration.append(integration)
                        if (step + 1) % 5 == 0:
                            milestones_pixels.append(pixels[0])
                            milestones_proprio.append(np.asarray(info["state"][:4], dtype=np.float32).copy())
                    else:
                        keys = ("reward", "done", "truncated", "success", "reach_distance", "tcp_before", "tcp_after", "state", "physical", "pixels", "environment_step", "policy_step")
                        require(all(row[k] == base_rows[step][k] for k in keys), f"Alias outcome mismatch candidate{candidate}/{arm}/control{step+1}")
                        require(np.array_equal(pixels, base_frames[step]), "Alias pixels differ")
                        require(all(np.array_equal(state[k], v) for k, v in base_phys[step].items()), "Alias physical state differs")
                        require(np.array_equal(integration, base_integration[step]), "Alias full integration state differs")
                    rows.append(row)
            if candidate == 0 and arm_index == 0:
                report["snapshot"]["historical_source_prefix_verified"] = True
                write(out / "snapshot_metadata.json", report["snapshot"])
            report["replays"].append({"candidate": candidate, "representation": arm, "controls": 30,
                                       "exact_to_original": True if arm_index else None})
            persist()
        def utility_at(n):
            rows = base_rows[:n]
            return {"controls": n, "terminal_native_reward": rows[-1]["reward"],
                    "summed_native_reward": sum(r["reward"] for r in rows),
                    "final_reach_distance": rows[-1]["reach_distance"],
                    "final_success": bool(rows[-1]["success"]), "ever_success": any(r["success"] for r in rows)}
        utility = {"candidate": candidate, "primary30": utility_at(30), "native_execution_prefix15": utility_at(15)}
        utilities.append(utility)
        gt_pixels.append(np.stack(milestones_pixels)); gt_proprio.append(np.stack(milestones_proprio))
        np.savez_compressed(out / f"candidate{candidate}" / "ground_truth.npz",
                            pixels=gt_pixels[-1], proprio=gt_proprio[-1],
                            full_state=np.asarray([r["state"] for r in base_rows]),
                            full_integration_state=np.stack(base_integration),
                            rewards=np.asarray([r["reward"] for r in base_rows]))
        report["utilities"] = utilities; persist()
        print(json.dumps({"phase": "replay", **utility}), flush=True)
    return np.stack(gt_pixels), np.stack(gt_proprio), utilities


def run_forecasts(cfg, norms, cached, gt_pixels, gt_proprio, utilities, out, report, persist, torch, np):
    from pilot_dino_metaworld_clean_gate import load_model, MODEL, DINO_WEIGHTS
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    original_load = torch.load
    def counted_load(path, *args, **kwargs):
        key = str(Path(path).resolve())
        require(key in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())), "Unexpected checkpoint")
        require(key not in report["checkpoint_loads"], "Checkpoint loaded more than once")
        report["checkpoint_loads"][key] = 1
        return original_load(path, *args, **kwargs)
    with patch.object(torch, "load", side_effect=counted_load):
        model, pre, loading = load_model(cfg, "cpu")
    report["model_load"] = loading
    require(len(report["checkpoint_loads"]) == 2, "Expected exactly two existing checkpoint loads")
    def tick_predictor(*_):
        report["activity"]["predictor_forwards"] += 1
    hook = model.model.predictor.register_forward_pre_hook(tick_predictor)
    def encode(pixels, proprio, act=True):
        report["activity"]["native_encodes"] += 1
        return model.encode({"visual": torch.from_numpy(pixels), "proprio": torch.from_numpy(proprio)}, act=act)
    z_init = encode(cached["initial_pixels"][None], cached["initial_proprio"][None])
    z_goal = encode(cached["goal_pixels"][None], cached["goal_proprio"][None], act=False)
    gt = []
    for h in range(6):
        gt.append(encode(gt_pixels[:, h, None], gt_proprio[:, h, None]))
    truth = {key: torch.stack([z[key][:, 0] for z in gt]) for key in ("visual", "proprio")}
    objective = ReprTargetDistMPCObjective(cfg, target_enc=z_goal, **cfg.planner.planning_objective)
    bank_actions = norms[0].reshape(5, 6, 20).permute(1, 0, 2).contiguous()
    results, saved, predictions = {}, {}, {}
    tol = 1e-6
    rewards = [x["primary30"]["terminal_native_reward"] for x in utilities]
    competent = None
    for key in truth:
        saved["ground_truth_" + key] = truth[key].cpu().numpy()
        saved["initial_" + key] = z_init[key].cpu().numpy()
        saved["goal_" + key] = z_goal[key].cpu().numpy()
    with torch.inference_mode():
        for index, arm in enumerate(ARMS):
            actions = norms[index].reshape(5, 6, 20).permute(1, 0, 2).contiguous()
            start = time.monotonic()
            report["activity"]["unroll_calls"] += 1
            predicted = model.unroll(z_init, act_suffix=actions)
            costs = objective(predicted, bank_actions)
            require(costs.shape == (5,) and torch.isfinite(costs).all(), "Invalid native objective output")
            per_error = {}
            for key in truth:
                require(predicted[key].shape[:2] == (7, 5) and torch.isfinite(predicted[key]).all(), "Invalid forecast shape/value")
                error = (predicted[key][1:] - truth[key]).square().mean(dim=tuple(range(2, predicted[key].ndim)))
                per_error[key] = error.cpu().numpy()
                saved[arm + "_" + key] = predicted[key].cpu().numpy().copy()
                require(np.array_equal(saved["initial_" + key], z_init[key].cpu().numpy()), "Initial encoding mutated")
            per_error["weighted"] = per_error["visual"] + .1 * per_error["proprio"]
            cost_list = costs.cpu().tolist()
            order = sorted(range(5), key=lambda i: (cost_list[i], i))
            results[arm] = {"seconds": time.monotonic() - start, "native_costs": cost_list,
                            "ranking": order, "selected_candidate": order[0],
                            "selected_true_utility": utilities[order[0]],
                            "cost_min_exact_ties": sum(v == min(cost_list) for v in cost_list),
                            "forecast_error": {k: {"per_horizon_candidate": v.tolist(), "final_per_candidate": v[-1].tolist(), "mean_over_horizons_per_candidate": v.mean(0).tolist()} for k, v in per_error.items()}}
            predictions[arm] = predicted
            report["forecasts"] = results; persist()
            print(json.dumps({"phase": "forecast", "arm": arm, "costs": cost_list, "selected": order[0]}), flush=True)
            if arm == "original":
                clean_reward = rewards[order[0]]
                competent = bool(clean_reward >= max(rewards) - tol and clean_reward > rewards[3] + tol and clean_reward > rewards[4] + tol)
                report["clean_competence_checked_before_alias_forecasts"] = competent
                persist()
                if not competent:
                    break
    hook.remove()
    for arm in list(predictions)[1:]:
        for key in truth:
            require(torch.equal(predictions[arm][key][:, 3], predictions["original"][key][:, 3]), "Unchanged zero candidate forecast differs")
    np.savez_compressed(out / "forecasts_and_encodings.npz", **saved)
    selections = {arm: result["selected_candidate"] for arm, result in results.items()}
    clean = rewards[selections["original"]]
    inflated = rewards[selections["inflated_saturated_xyz"]] if competent else None
    canonical = rewards[selections["canonical_saturated_xyz"]] if competent else None
    harm = bool(selections["original"] != selections["inflated_saturated_xyz"] and inflated < clean - tol) if competent else None
    gate = {"clean_competence_passed": bool(competent), "bank_terminal_rewards": rewards,
            "bank_best_terminal_reward_ties_within_tolerance": sum(v >= max(rewards) - tol for v in rewards),
            "inflated_selected_index_changed": selections["original"] != selections["inflated_saturated_xyz"] if competent else None,
            "harm_signal_conditioned_on_clean": harm,
            "canonical_recovery_if_harm": bool(canonical >= clean - tol) if harm else None,
            "inflated_minus_clean_terminal_reward": inflated - clean if competent else None,
            "canonical_minus_clean_terminal_reward": canonical - clean if competent else None,
            "canonical_minus_inflated_terminal_reward": canonical - inflated if competent else None,
            "canonical_controls": "Canonical inputs from Clean/Inflated are identical by construction; if Clean passes, both share one measured canonical forecast batch with actual replay-derived utility. Otherwise no canonical forecast was executed.",
            "unchanged_zero_forecast_exact": True if competent else None,
            "skipped_forecasts": [] if competent else list(ARMS[1:]),
            "stop_reason": None if competent else "Frozen Clean bank competence gate failed; no alias/canonical model calls",
            "tolerance": tol, "tolerance_is_practical_harm_threshold": False,
            "scope": "One root, added H6=30-control open-loop bank selector; not native CEM (which executes15 before replanning), not independent generalization or statistical significance."}
    report["gate"] = gate
    return gate


def main(config_path, override_out):
    protocol = json.loads(config_path.read_text())
    out = (override_out or ROOT / protocol["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    report = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path.resolve()), "activity": {"environment_steps": 0, "unroll_calls": 0, "predictor_forwards": 0, "native_encodes": 0, "network_attempts": 0, "planner_calls": 0, "training_updates": 0},
              "replays": [], "checkpoint_loads": {}, "gate": None}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        write(out / "summary.json", report)
    def timeout_handler(signum, frame):
        raise TimeoutError("Frozen wall-clock limit reached")
    def no_network(*args, **kwargs):
        report["activity"]["network_attempts"] += 1
        raise RuntimeError("Network access prohibited")
    signal.signal(signal.SIGALRM, timeout_handler); signal.signal(signal.SIGTERM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, protocol["runtime"]["wall_limit_seconds"])
    env = None
    persist()
    try:
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            import numpy as np
            import torch
            import mujoco
            from pilot_dino_metaworld_clean_gate import cfg_for
            from app.plan_common.datasets import get_data_stats
            from app.plan_common.datasets.preprocessor import Preprocessor
            from evals.simu_env_planning.envs.init import make_env
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            require(mujoco.__version__ == "3.3.0" and str(ROOT / ".deps/dino_mujoco330") in mujoco.__file__, "Wrong MuJoCo runtime")
            require(not torch.cuda.is_initialized(), "CUDA initialized")
            torch.set_num_threads(4)
            random.seed(1); np.random.seed(1); torch.manual_seed(1)
            report["runtime"] = {"interpreter": sys.executable, "python": sys.version, "torch": torch.__version__, "mujoco": mujoco.__version__, "mujoco_file": mujoco.__file__, "renderer": "osmesa", "device": "cpu"}
            stats = get_data_stats("metaworld")
            pre = Preprocessor(**{k: torch.tensor(stats[k]) for k in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")}, transform=None)
            norms, requests, historical, report["bank"] = make_bank(protocol, pre, torch, np)
            np.savez_compressed(out / "frozen_bank.npz", normalized=norms.numpy(), requested_physical=requests.numpy())
            source = ROOT / protocol["root"]["directory"]
            keys = ["sim_" + n for n in SIM_FIELDS] + ["sim_time", "model_site_pos", "target", "rand_vec", "previous_observation", "elapsed", "initial_pixels", "initial_proprio", "initial_state", "goal_pixels", "goal_proprio"]
            with np.load(source / "setup.npz", allow_pickle=False) as archive:
                cached = {k: archive[k].copy() for k in keys}
            cfg = cfg_for(out, "cpu")
            report["status"] = "reconstructing_root"; persist()
            env = make_env(cfg)
            restore, report["snapshot"] = reconstruct_and_snapshot(env, cached, mujoco, torch, np)
            write(out / "snapshot_metadata.json", report["snapshot"])
            report["status"] = "replaying"; persist()
            pixels, proprio, utilities = replay_bank(env, restore, requests, historical, out, report, persist, torch, np)
            env.close(); env = None
            report["status"] = "loading_and_forecasting"; persist()
            run_forecasts(cfg, norms, cached, pixels, proprio, utilities, out, report, persist, torch, np)
            require(not torch.cuda.is_initialized(), "CUDA initialized")
            report["cuda_initialized"] = False
            report["status"] = "completed_cpu_fixed_bank_gate"
    except BaseException as error:
        report.update(status="failed_closed_cpu_fixed_bank_gate", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            env.close()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "gate": report["gate"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bootstrap()
    main(args.config.resolve(), args.out)
