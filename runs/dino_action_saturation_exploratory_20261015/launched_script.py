#!/usr/bin/env python3
"""Exploratory cached DINO alias forecasts; no environment or new encodings."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dino_action_saturation_exploratory_20261015.json"
ARMS = ("original", "inflated_saturated_xyz", "canonical_saturated_xyz")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(config_path, override_out):
    protocol = json.loads(config_path.read_text())
    out = (override_out or ROOT / protocol["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    source = ROOT / protocol["source_run"]
    started = time.monotonic()
    counts = dict(unroll_calls=0, runtime_replay_unrolls=0, new_alias_unrolls=0,
                  predictor_forwards=0, native_encodes=0, visual_encoder_forwards=0,
                  proprio_encoder_forwards=0, environment_steps=0, planner_calls=0,
                  network_attempts=0, training_updates=0)
    report = {"status": "starting_exploratory_continuation", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path), "source_run": str(source), "disclosure": protocol["disclosure"],
              "activity": counts, "checkpoint_loads": {}, "conditions": {}, "runtime_replay": None,
              "prior_oracle_gate_passed": False, "new_passing_threshold": None,
              "scope": protocol["analysis_boundary"]}
    forecasts = {}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def stop(signum, frame):
        raise TimeoutError("Fixed180-second CPU continuation limit reached")
    def forbidden(name):
        def call(*args, **kwargs):
            counts[name] += 1
            raise RuntimeError("Forbidden continuation operation: " + name)
        return call
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, 180)
    persist()
    try:
        previous = json.loads((source / "summary.json").read_text())
        require(previous["status"] == "completed_cpu_fixed_bank_gate", "Predecessor not completed")
        require(previous["gate"]["clean_competence_passed"] is False, "Expected unchanged failed predecessor gate")
        require(previous["activity"]["environment_steps"] == 450 and len(previous["replays"]) == 15, "Incomplete prior physical replays")
        require(previous["snapshot"]["historical_source_prefix_verified"] and previous["bank"]["effective_xyz_and_gripper_exact"], "Prior equivalence/historical gate missing")
        require(all(r["exact_to_original"] for r in previous["replays"] if r["representation"] != "original"), "Prior alias replay was not exact")
        require(list(previous["forecasts"]) == ["original"], "Expected only original forecasts in predecessor")
        report["previous_gate_unchanged"] = previous["gate"]
        report["candidate_names"] = previous["bank"]["candidate_names"]
        report["candidate_real_utilities"] = previous["utilities"]

        import numpy as np
        import torch
        from tensordict import TensorDict
        torch.set_num_threads(4)
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
        report["runtime"] = {"interpreter": sys.executable, "python": sys.version, "torch": torch.__version__,
                              "torch_path": torch.__file__, "threads": torch.get_num_threads(), "device": "cpu"}
        with np.load(source / "frozen_bank.npz", allow_pickle=False) as archive:
            normalized = archive["normalized"].copy()
            requested = archive["requested_physical"].copy()
        require(normalized.shape == requested.shape == (3, 5, 30, 4), "Frozen bank schema changed")
        require(normalized.dtype == requested.dtype == np.float32 and np.isfinite(normalized).all() and np.isfinite(requested).all(), "Frozen bank dtype/value changed")
        with np.load(source / "forecasts_and_encodings.npz", allow_pickle=False) as archive:
            cached = {k: archive[k].copy() for k in archive.files}
        require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in cached.values()), "Cached latent dtype/value changed")
        z_init = TensorDict({k: torch.from_numpy(cached["initial_" + k]).clone() for k in ("visual", "proprio")})
        z_goal = TensorDict({k: torch.from_numpy(cached["goal_" + k]).clone() for k in ("visual", "proprio")})
        truth = {k: torch.from_numpy(cached["ground_truth_" + k]) for k in ("visual", "proprio")}
        require(z_init["visual"].shape[:2] == (1, 1) and z_goal["visual"].shape[:2] == (1, 1), "Expected original single-frame context/goal")
        require(all(t.shape[:2] == (6, 5) for t in truth.values()), "Expected six real future endpoints and five candidates")
        norms = torch.from_numpy(normalized)
        original_actions = norms[0].reshape(5, 6, 20).permute(1, 0, 2).contiguous()
        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, MODEL, DINO_WEIGHTS
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
        original_load = torch.load
        def load_once(path, *args, **kwargs):
            key = str(Path(path).resolve())
            require(key in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and key not in report["checkpoint_loads"], "Unexpected/repeated checkpoint load")
            report["checkpoint_loads"][key] = 1
            return original_load(path, *args, **kwargs)

        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=forbidden("network_attempts")))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=forbidden("network_attempts")))
            guards.enter_context(patch.object(torch, "load", side_effect=load_once))
            cfg = cfg_for(out, "cpu")
            report["status"] = "loading_existing_checkpoint"; persist()
            model, pre, report["strict_load"] = load_model(cfg, "cpu")
            require(len(report["checkpoint_loads"]) == 2, "Expected exactly two checkpoint files")
            guards.enter_context(patch.object(model, "encode", side_effect=forbidden("native_encodes")))
            hooks = [model.model.encoder.register_forward_pre_hook(forbidden("visual_encoder_forwards")),
                     model.model.proprio_encoder.register_forward_pre_hook(forbidden("proprio_encoder_forwards")),
                     model.model.predictor.register_forward_pre_hook(lambda *_: counts.__setitem__("predictor_forwards", counts["predictor_forwards"] + 1))]
            for hook in hooks:
                guards.callback(hook.remove)
            denormalized = pre.denormalize_actions(norms)
            require(np.array_equal(denormalized.numpy(), requested), "Native denormalization differs from frozen execution requests")
            sat = torch.zeros_like(norms[0], dtype=torch.bool); sat[:, :, :3] = denormalized[0, :, :, :3].abs() > 1
            for i in range(3):
                require(torch.equal(norms[i][~sat], norms[0][~sat]), "Untouched normalized coordinates changed")
                require(torch.equal(denormalized[i, :, :, :3].clamp(-1, 1), denormalized[0, :, :, :3].clamp(-1, 1)), "Effective XYZ changed")
                require(torch.equal(denormalized[i, :, :, 3], denormalized[0, :, :, 3]), "Gripper changed")
            require(torch.equal(norms[0, 3], norms[1, 3]) and torch.equal(norms[0, 3], norms[2, 3]), "Zero request differs")
            objective = ReprTargetDistMPCObjective(cfg, target_enc=z_goal, **cfg.planner.planning_objective)
            with torch.inference_mode():
                for index, arm in enumerate(ARMS):
                    report["status"] = "forecasting_" + arm; persist()
                    actions = norms[index].reshape(5, 6, 20).permute(1, 0, 2).contiguous()
                    tick = time.monotonic()
                    counts["unroll_calls"] += 1
                    counts["runtime_replay_unrolls" if index == 0 else "new_alias_unrolls"] += 1
                    prediction = model.unroll(z_init, act_suffix=actions)
                    costs = objective(prediction, original_actions)
                    require(costs.shape == (5,) and torch.isfinite(costs).all(), "Invalid native costs")
                    for key in truth:
                        require(prediction[key].shape == cached["original_" + key].shape and torch.isfinite(prediction[key]).all(), "Invalid forecast schema/value")
                        forecasts[arm + "_" + key] = prediction[key].cpu().numpy().copy()
                    np.savez_compressed(out / "condition_forecasts.npz", **forecasts)
                    if index == 0:
                        checks = {key: np.array_equal(forecasts[arm + "_" + key], cached["original_" + key]) for key in truth}
                        checks["costs"] = np.array_equal(costs.cpu().numpy(), np.asarray(previous["forecasts"]["original"]["native_costs"], dtype=np.float32))
                        report["runtime_replay"] = {"exact": all(checks.values()), "checks": checks}
                        persist()
                        require(all(checks.values()), "Original CPU runtime replay is not exact; no alias call permitted")
                    else:
                        require(all(np.array_equal(forecasts[arm + "_" + k][:, 3], forecasts["original_" + k][:, 3]) for k in truth), "Unchanged zero candidate forecast differs")
                    errors = {}
                    divergence = {}
                    for key in truth:
                        dims = tuple(range(2, prediction[key].ndim))
                        errors[key] = (prediction[key][1:] - truth[key]).square().mean(dim=dims).cpu().numpy()
                        delta = prediction[key][1:] - torch.from_numpy(forecasts["original_" + key][1:])
                        divergence[key] = delta.square().mean(dim=dims).cpu().numpy()
                    errors["weighted"] = errors["visual"] + .1 * errors["proprio"]
                    cost_list = costs.cpu().tolist()
                    ranking = sorted(range(5), key=lambda i: (cost_list[i], i))
                    report["conditions"][arm] = {
                        "seconds": time.monotonic() - tick, "native_costs": cost_list, "ranking": ranking,
                        "selected_candidate": ranking[0], "selected_real_utility": previous["utilities"][ranking[0]],
                        "cost_min_exact_ties": sum(x == min(cost_list) for x in cost_list),
                        "forecast_errors": {k: {"per_horizon_candidate": v.tolist(), "final_per_candidate": v[-1].tolist(), "mean_over_horizons_per_candidate": v.mean(0).tolist()} for k, v in errors.items()},
                        "forecast_mse_difference_vs_original": {k: v.tolist() for k, v in divergence.items()},
                        "zero_forecast_exact_to_original": True
                    }
                    require(all(np.array_equal(z_init[k].numpy(), cached["initial_" + k]) and np.array_equal(z_goal[k].numpy(), cached["goal_" + k]) for k in truth), "Frozen context/goal mutated")
                    persist()
                    print(json.dumps({"arm": arm, "native_costs": cost_list, "selected": ranking[0], "terminal_reward30": previous["utilities"][ranking[0]]["primary30"]["terminal_native_reward"]}), flush=True)

        contrasts = {}
        for right, left in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
            a, b = report["conditions"][left], report["conditions"][right]
            pairs_changed = sum((a["ranking"].index(i) < a["ranking"].index(j)) != (b["ranking"].index(i) < b["ranking"].index(j)) for i in range(5) for j in range(i + 1, 5))
            effect = {"selected_candidate_changed": a["selected_candidate"] != b["selected_candidate"], "pairwise_order_changes": pairs_changed, "candidate_pairs": 10}
            for endpoint in ("primary30", "native_execution_prefix15"):
                ua, ub = a["selected_real_utility"][endpoint], b["selected_real_utility"][endpoint]
                effect[endpoint] = {k: float(ub[k]) - float(ua[k]) for k in ("terminal_native_reward", "summed_native_reward", "final_reach_distance", "final_success", "ever_success")}
            contrasts[right + "_minus_" + left] = effect
        report["signed_descriptive_contrasts"] = contrasts
        report["cuda_initialized"] = torch.cuda.is_initialized()
        require(not report["cuda_initialized"], "CUDA initialized")
        report["status"] = "completed_exploratory_cached_alias_forecasts"
        report["interpretation"] = "Previous oracle gate remains failed. Signed utility contrasts are descriptive within this unchanged bank; no attack success rate, independent replication, new passing threshold or complete native-CEM claim."
    except BaseException as error:
        report.update(status="failed_closed_exploratory_continuation", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "activity": counts, "contrasts": report["signed_descriptive_contrasts"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    main(args.config.resolve(), args.out)
