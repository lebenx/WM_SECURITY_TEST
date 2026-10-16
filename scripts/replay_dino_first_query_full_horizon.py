#!/usr/bin/env python3
"""Replay exactly two frozen query1 means for30 controls, CPU only."""
from __future__ import annotations

import argparse
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
CONFIG = ROOT / "configs/dino_first_query_full_horizon_replay_20261015.json"
ARMS = ("clean_identity_copy", "canonical_request")
PREFIX_FIELDS = ("action", "reward", "done", "truncated", "success", "reach_distance",
                 "tcp_before", "tcp_after", "state", "physical", "pixels", "environment_step", "policy_step")
HISTORICAL_PREFIX_SCOPE = "Exact controls1..15 for both fixed candidates against their respective source traces. Controls16..30 are newly measured open-loop outcomes, not a historical-prefix reproduction claim."


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utility(rows):
    return {"controls": len(rows), "terminal_native_reward": rows[-1]["reward"],
            "cumulative_native_reward": sum(r["reward"] for r in rows),
            "terminal_reach_distance": rows[-1]["reach_distance"],
            "terminal_success": rows[-1]["success"], "ever_success": any(r["success"] for r in rows),
            "successful_controls": sum(r["success"] for r in rows),
            "first_success_control": next((r["policy_step"] for r in rows if r["success"]), None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    from probe_dino_action_saturation_gate import bootstrap, reconstruct_and_snapshot, SIM_FIELDS
    bootstrap()
    protocol = json.loads(args.config.read_text())
    require(protocol["status"] == "frozen_before_any_new_full_horizon_replay", "Protocol not frozen")
    require(protocol["root"]["physical_seed"] == 4 and protocol["candidates"]["order"] == list(ARMS), "Unregistered root/bank")
    require(protocol["runtime"]["maximum_explicit_environment_steps"] == 60
            and protocol["runtime"]["wall_limit_seconds"] == 300, "Budget changed")
    out, report_path = ROOT / protocol["out_dir"], ROOT / protocol["report"]
    require(not out.exists() and not report_path.exists(), "Refusing to overwrite existing evidence")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(args.config.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    summary = {"status": "starting_cpu_replay", "protocol": str(args.config.resolve()),
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "active": None,
               "candidate_order": list(ARMS), "candidates": [], "comparison": None,
               "activity": {"explicit_environment_steps": 0, "environment_instances": 0,
                            "model_calls": 0, "encodes": 0, "checkpoint_loads": 0, "planner_calls": 0,
                            "api_calls": 0, "gpu_calls": 0, "network_attempts": 0, "training_updates": 0,
                            "expert_actions_loaded": False, "source_files_rehashed": 0},
               "limits": [protocol["interpretation"],
                          "A cached-state reconstruction, not a new independent root. Constructor-internal initialization is not counted as explicit replay controls; no explicit reset/expert/warmup is performed.",
                          "trace_step computes per-step physical/pixel digests for exact historical comparison; no checkpoint or existing-file hashes are recomputed.",
                          "Partial execution is not a terminal task failure or a completed horizon comparison."]}
    def persist():
        summary["elapsed_seconds"] = time.monotonic() - started
        write(out / "summary.json", summary)
    def timeout_handler(*_):
        raise TimeoutError("Frozen300-second CPU replay limit reached")
    def denied(*_args, **_kwargs):
        summary["activity"]["network_attempts"] += 1
        raise RuntimeError("Network disabled")
    def no_checkpoint(*_args, **_kwargs):
        raise RuntimeError("No checkpoint or model loading is allowed")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 300)
    env, exit_code = None, 0
    persist()
    try:
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=denied))
            import numpy as np
            import torch
            import mujoco
            from PIL import Image
            from pilot_dino_metaworld_clean_gate import cfg_for, physical, trace_step
            from app.plan_common.datasets.preprocessor import Preprocessor
            from evals.simu_env_planning.envs.init import make_env
            guards.enter_context(patch.object(torch, "load", side_effect=no_checkpoint))
            guards.enter_context(patch.object(torch.hub, "load", side_effect=no_checkpoint))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=denied))
            require(mujoco.__version__ == "3.3.0" and not torch.cuda.is_initialized(), "Wrong runtime/CUDA initialized")
            torch.set_num_threads(4)
            random.seed(1); np.random.seed(1); torch.manual_seed(1)
            summary["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__,
                                  "numpy": np.__version__, "mujoco": mujoco.__version__,
                                  "renderer": "osmesa", "device": "cpu", "cuda_initialized": False}
            source = ROOT / protocol["root"]["source_run"]
            native = json.loads((source / "summary.json").read_text())
            ranking = json.loads((ROOT / protocol["known_before_execution"]["ranking_report"]).read_text())
            require(native["status"] == "completed_native_alias_pair" and native["setup"]["seed"] == 4,
                    "Source native root not complete/seed4")
            require(ranking["status"] == "completed_cached_two_mean_ranking"
                    and ranking["candidate_order"] == list(ARMS)
                    and ranking["cost_matrix"]["terminal_control_horizon"] == 30, "Wrong fixed ranking")
            require(ranking["cost_matrix"]["values"] == protocol["known_before_execution"]["cost_matrix_original_then_canonical"],
                    "Frozen forecast scores changed")
            selected = {name: ranking["rankings"][name]["selected_candidate_index"] for name in ("original", "canonical")}
            require(selected == {"original": 0, "canonical": 1}, "Frozen selection changed")
            summary["frozen_scoring"] = {"source": protocol["known_before_execution"]["ranking_report"],
                                         "cost_matrix": ranking["cost_matrix"], "selected_indices": selected,
                                         "new_objective_calls": 0}
            stats = native["strict_load"]["normalization"]
            pre = Preprocessor(**{k: torch.tensor(stats[k], dtype=torch.float32) for k in
                ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")}, transform=None)
            requests, means, historical = [], [], []
            for arm in ARMS:
                with np.load(source / arm / "query1_forecast_cache.npz", allow_pickle=False) as archive:
                    mean = archive["mean_original"].copy()
                require(mean.shape == (6, 1, 20) and mean.dtype == np.float32, "Frozen mean shape/dtype differs")
                raw = pre.denormalize_actions(torch.from_numpy(mean).reshape(30, 4))
                with (source / arm / "trace.jsonl").open() as handle:
                    old = [json.loads(line) for line in islice(handle, 15)]
                require(len(old) == 15 and torch.equal(raw[:15], torch.tensor([r["action"] for r in old], dtype=torch.float32)),
                        "Decoded full mean does not reproduce historical15 controls")
                require(bool(torch.isfinite(raw).all()), "Nonfinite fixed actions")
                requests.append(raw); means.append(mean); historical.append(old)
            np.savez_compressed(out / "physical_actions.npz", mean_original=np.stack(means),
                                physical_actions=torch.stack(requests).numpy(),
                                action_mean=pre.action_mean.numpy(), action_std=pre.action_std.numpy())
            keys = ["sim_" + n for n in SIM_FIELDS] + ["sim_time", "model_site_pos", "target", "rand_vec",
                    "previous_observation", "elapsed", "initial_pixels", "initial_proprio", "initial_state"]
            with np.load(ROOT / protocol["root"]["setup"], allow_pickle=False) as archive:
                cached = {key: archive[key].copy() for key in keys}
            with np.load(ROOT / protocol["root"]["snapshot"], allow_pickle=False) as archive:
                snapshot = {key: archive[key].copy() for key in archive.files if key != "torch_cuda_rng"}
            cfg = cfg_for(out, "cpu")
            summary["status"] = "reconstructing_cached_seed4_root"; persist()
            env = make_env(cfg); summary["activity"]["environment_instances"] += 1
            raw_env = env.proprio_env.unwrapped
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            raw_env.model.site_pos[:] = snapshot["site_pos"]
            mujoco.mj_setState(raw_env.model, raw_env.data, snapshot["integration"], spec)
            torch.set_rng_state(torch.from_numpy(snapshot["torch_cpu_rng"]))
            restore, summary["reconstruction"] = reconstruct_and_snapshot(env, cached, mujoco, torch, np)
            integration = np.empty(mujoco.mj_stateSize(raw_env.model, spec), dtype=np.float64)
            mujoco.mj_getState(raw_env.model, raw_env.data, integration, spec)
            require(np.array_equal(integration, snapshot["integration"]), "Source full integration reconstruction differs")
            require(np.array_equal(raw_env.model.site_pos, snapshot["site_pos"]), "Source site positions differ")
            current_physical = physical(env)
            require(all(np.array_equal(v, snapshot["physical_" + k]) for k, v in current_physical.items()),
                    "Source full physical snapshot differs")
            summary["reconstruction"].update(source_integration_exact=True, source_physical_exact=True,
                                             source_site_positions_exact=True, expert_actions_loaded=False,
                                             historical_source_prefix_scope=HISTORICAL_PREFIX_SCOPE)
            summary["status"] = "replaying_frozen_full_means"; persist()
            for index, arm in enumerate(ARMS):
                restore()
                directory = out / arm; directory.mkdir()
                rows = []
                record = {"index": index, "name": arm, "status": "running", "controls": 0,
                          "historical_prefix_exact_steps": 0, "trace": str((directory / "trace.jsonl").relative_to(ROOT))}
                summary["candidates"].append(record)
                summary["active"] = {"candidate": index, "name": arm, "controls": 0}; persist()
                with (directory / "trace.jsonl").open("x") as handle:
                    for step in range(30):
                        require(summary["activity"]["explicit_environment_steps"] < 60, "Explicit control budget exceeded")
                        # Count a dispatched step even if the simulator raises.
                        summary["activity"]["explicit_environment_steps"] += 1
                        visual, info, row = trace_step(env, requests[index][step].clone(), 1, step)
                        row.update(replay_candidate=index, replay_candidate_name=arm)
                        handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                        rows.append(row); record["controls"] = len(rows)
                        summary["active"]["controls"] = len(rows)
                        require(not raw_env._did_see_sim_exception and not row["done"] and not row["truncated"],
                                "Unexpected simulator exception or early termination")
                        if step < 15:
                            mismatch = [k for k in PREFIX_FIELDS if row[k] != historical[index][step][k]]
                            record["last_prefix_mismatches"] = mismatch
                            require(not mismatch, f"Historical prefix mismatch candidate{index}/control{step+1}: {mismatch}")
                            record["historical_prefix_exact_steps"] += 1
                        if step + 1 == 15:
                            summary["reconstruction"]["historical_source_prefix_verified"] = (
                                len(summary["candidates"]) == len(ARMS)
                                and all(c["historical_prefix_exact_steps"] == 15
                                        and not c["last_prefix_mismatches"] for c in summary["candidates"]))
                            record["prefix15"] = utility(rows)
                            np.savez_compressed(directory / "endpoint15.npz", pixels=visual.cpu().numpy().copy(),
                                                proprio=np.asarray(info["state"][:4], dtype=np.float32).copy()[None],
                                                state=np.asarray(info["state"]).copy(), tcp=physical(env)["tcp"],
                                                target=cached["target"], control_offset=np.array(15))
                            record["endpoint15"] = str((directory / "endpoint15.npz").relative_to(ROOT))
                            print(json.dumps({"candidate": index, "prefix15_exact": True, "utility": record["prefix15"]}), flush=True)
                        if (step + 1) % 5 == 0:
                            persist()
                pixels = visual.cpu().numpy().copy()
                proprio = np.asarray(info["state"][:4], dtype=np.float32).copy()[None]
                np.savez_compressed(directory / "endpoint30.npz", pixels=pixels, proprio=proprio,
                                    state=np.asarray(info["state"]).copy(), tcp=physical(env)["tcp"], target=cached["target"],
                                    control_offset=np.array(30))
                Image.fromarray(pixels[0].transpose(1, 2, 0)).save(directory / "endpoint30.png")
                record.update(status="completed30", utility30=utility(rows),
                              endpoint30=str((directory / "endpoint30.npz").relative_to(ROOT)),
                              endpoint30_image=str((directory / "endpoint30.png").relative_to(ROOT)))
                print(json.dumps({"candidate": index, "utility30": record["utility30"]}), flush=True)
                persist()
            require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
            original = summary["candidates"][selected["original"]]["utility30"]
            canonical = summary["candidates"][selected["canonical"]]["utility30"]
            delta = canonical["terminal_native_reward"] - original["terminal_native_reward"]
            summary["comparison"] = {
                "score_and_true_outcome_control_horizon": 30, "selection_indices_frozen": selected,
                "primary_canonical_selected_minus_original_selected_terminal_reward": delta,
                "selected_terminal_reward_decreases": delta < 0,
                "selected_terminal_reward_ties": delta == 0,
                "selected_terminal_reward_increases": delta > 0,
                "cumulative_reward30_difference": canonical["cumulative_native_reward"] - original["cumulative_native_reward"],
                "terminal_distance_difference": canonical["terminal_reach_distance"] - original["terminal_reach_distance"],
                "terminal_success_difference": int(canonical["terminal_success"]) - int(original["terminal_success"]),
                "scope": "Measured two-candidate full-horizon selection consequence; no complete-population CEM or closed-loop causal mediation claim."}
            summary["status"] = "completed_two_frozen_mean_full_horizon_replays"
    except BaseException as error:
        exit_code = 1
        summary.update(status="resource_incomplete" if isinstance(error, TimeoutError) else "invalid_reconstruction_or_prefix_or_runtime",
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            env.close()
        summary["last_active"] = summary["active"]; summary["active"] = None
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        report = dict(summary)
        report["source_summary"] = str((out / "summary.json").relative_to(ROOT))
        report["reproduction"] = "Frozen single-attempt replay; output paths are exclusive. Read summary/traces for verification; do not silently rerun or overwrite."
        write(report_path, report)
    print(json.dumps({"status": summary["status"], "comparison": summary["comparison"],
                      "activity": summary["activity"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
