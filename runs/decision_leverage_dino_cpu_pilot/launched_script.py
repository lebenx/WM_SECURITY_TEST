#!/usr/bin/env python3
"""Prospective fixed-bank DINO oracle-feature diagnostic on fresh physical roots.

This is an added open-loop ranker, not native CEM or a deployment defense.
Calibration is seed1000 only. Confirmation is the frozen1001..1030 list.
No alias operations, model training, GPU, network, or outcome-based tuning.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/decision_leverage_dino_e1.json"
LAMBDAS = [0., .25, .5, .75, 1.]
FEATURES = ("visual", "proprio")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def utilities(rows):
    result = {}
    for horizon in (15, 30):
        t = rows[:horizon]
        require(len(t) == horizon, "Incomplete physical utility trajectory")
        result[f"h{horizon}"] = {
            "terminal_reward": t[-1]["reward"], "cumulative_reward": sum(x["reward"] for x in t),
            "negative_terminal_distance": -t[-1]["reach_distance"],
            "terminal_success": int(t[-1]["success"]), "ever_success": int(any(x["success"] for x in t)),
            "first_success_control": next((i + 1 for i, x in enumerate(t) if x["success"]), None)}
    return result


def summarize_conditions(scores, errors, utility):
    truth_selected = min(range(5), key=lambda i: (scores["1.0"][i], i))
    result = {}
    for label, costs in scores.items():
        ranking = sorted(range(5), key=lambda i: (costs[i], i))
        selected = ranking[0]
        metrics = {}
        for horizon in ("h15", "h30"):
            metrics[horizon] = {}
            for key in utility[0][horizon]:
                if key == "first_success_control":
                    continue
                values = [u[horizon][key] for u in utility]
                mis = max(values) - values[truth_selected]
                pred = values[truth_selected] - values[selected]
                regret = max(values) - values[selected]
                require(abs(mis + pred - regret) < 1e-10, "Regret identity failed")
                metrics[horizon][key] = {"Lev": max(values) - min(values), "Mis": mis, "Pred": pred,
                                         "regret": regret, "selected_utility": values[selected],
                                         "true_future_selected_utility": values[truth_selected],
                                         "uniform_expected_utility": sum(values) / 5,
                                         "fixed_candidate_utilities": values}
        result[label] = {"lambda": float(label), "native_costs": costs, "ranking": ranking,
                         "selected_candidate": selected, "exact_minimum_ties": sum(c == min(costs) for c in costs),
                         "cost_margin": costs[ranking[1]] - costs[ranking[0]],
                         "terminal_forecast_errors": errors[label], "metrics": metrics}
    return result


def tensor_analysis(arrays, torch, np, objective=None, actions=None):
    """Pure cached arithmetic; B=5 and historical FP32 native reduction preserved."""
    from tensordict import TensorDict
    truth = {k: torch.from_numpy(arrays["truth_" + k]) for k in FEATURES}
    prediction = {k: torch.from_numpy(arrays["forecast_" + k]) for k in FEATURES}
    goal = {k: torch.from_numpy(arrays["goal_" + k]) for k in FEATURES}
    scores, errors = {}, {}
    for lam in LAMBDAS:
        label = str(lam)
        mixed = {k: (prediction[k][1:].clone() if lam == 0 else truth[k].clone() if lam == 1
                     else (1 - lam) * prediction[k][1:] + lam * truth[k]) for k in FEATURES}
        components = {k: (goal[k] - mixed[k]).pow(2).mean(dim=tuple(range(2, mixed[k].ndim)))[-1]
                      for k in FEATURES}
        costs = components["visual"] + .1 * components["proprio"]
        require(torch.isfinite(costs).all() and tuple(costs.shape) == (5,), "Nonfinite/wrong native score shape")
        if objective is not None:
            native = objective(TensorDict(mixed), actions)
            require(torch.equal(native, costs), "Native objective vs recorded FP32 expression differs")
            if lam == 0:
                require(torch.equal(objective(TensorDict(prediction), actions), costs),
                        "Initial context removal changes native terminal-score numerics")
        scores[label] = costs.tolist()
        errs = {k: (mixed[k][-1].double() - truth[k][-1].double()).square()
                .mean(dim=tuple(range(1, mixed[k][-1].ndim))).tolist() for k in FEATURES}
        errs["weighted"] = [v + .1 * p for v, p in zip(errs["visual"], errs["proprio"])]
        errs["mean_weighted"] = sum(errs["weighted"]) / 5
        errors[label] = errs
    require(errors["1.0"]["mean_weighted"] == 0, "Oracle substitution must be exact")
    for left, right in zip(LAMBDAS, LAMBDAS[1:]):
        require(all(b <= a + 1e-12 for a, b in zip(errors[str(left)]["weighted"], errors[str(right)]["weighted"])),
                "Oracle interpolation increased actual future error")
    return scores, errors


def verify_root(directory, torch, np):
    report = json.loads((directory / "result.json").read_text())
    require(report["status"] == "completed", "Cannot resume unfinished root: " + str(directory))
    for name, expected in report["artifact_sha256"].items():
        require(digest(directory / name) == expected, "Cached root artifact changed: " + name)
    utility = []
    for candidate in range(5):
        trace = [json.loads(x) for x in (directory / f"candidate{candidate}/trace.jsonl").read_text().splitlines()]
        require([r["policy_step"] for r in trace] == list(range(1, 31)), "Candidate trace gap")
        utility.append(utilities(trace))
    with np.load(directory / "features.npz", allow_pickle=False) as cache:
        arrays = {k: cache[k].copy() for k in cache.files}
        scores, errors = tensor_analysis(arrays, torch, np)
    with np.load(directory / "forecast_commitment.npz", allow_pickle=False) as committed:
        require(all(np.array_equal(arrays[k], committed[k]) for k in committed.files), "Committed forecast/context/goal changed")
    require(utility == report["candidate_utilities"], "Cached utility differs from actual traces")
    require(summarize_conditions(scores, errors, utility) == report["conditions"], "Cached oracle-score analysis differs")
    return report


def execute_root(seed, out, cfg, model, pre, norm, requests, torch, np, mujoco, overall):
    from pilot_dino_metaworld_clean_gate import setup_episode, trace_step, physical, array_digest
    from probe_dino_action_saturation_gate import reconstruct_and_snapshot
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    directory = out / f"seed{seed}"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"status": "starting", "seed": seed, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "activity": {"expert_controls": 0, "warmup_controls": 0, "candidate_controls": 0,
                           "encode_calls": 0, "unroll_calls": 0}, "candidate_utilities": []}
    env = None
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        write(directory / "result.json", report)
    def encode(pixels, proprio, act=True):
        report["activity"]["encode_calls"] += 1
        return model.encode({"visual": torch.from_numpy(pixels), "proprio": torch.from_numpy(proprio)}, act=act)
    persist()
    try:
        env, _, _, setup = setup_episode(cfg, seed, directory / "setup")
        report["activity"].update(expert_controls=100, warmup_controls=1)
        report["setup"] = setup
        with np.load(directory / "setup/setup.npz", allow_pickle=False) as cached:
            root = {k: cached[k].copy() for k in cached.files}
        restore, metadata = reconstruct_and_snapshot(env, root, mujoco, torch, np)
        metadata["source"] = "fresh native one-warmup root after independent expert goal construction"
        metadata.pop("historical_source_prefix_verified", None)
        report["restore"] = metadata
        report["root_goal_fingerprint"] = hashlib.sha256(root["sim_qpos"].tobytes() + root["sim_qvel"].tobytes()
                                                         + root["target"].tobytes() + root["goal_pixels"].tobytes()).hexdigest()
        report["status"] = "forecast_before_candidate_replays"; persist()
        z_init = encode(root["initial_pixels"][None], root["initial_proprio"][None])
        z_goal = encode(root["goal_pixels"][None], root["goal_proprio"][None], act=False)
        frozen = {"initial_" + k: z_init[k].numpy().copy() for k in FEATURES}
        frozen.update({"goal_" + k: z_goal[k].numpy().copy() for k in FEATURES})
        actions = norm.reshape(5, 6, 20).permute(1, 0, 2).contiguous()
        forecast_tick = time.monotonic()
        report["activity"]["unroll_calls"] += 1
        forecast = model.unroll(z_init, act_suffix=actions.clone())
        report["forecast_seconds"] = time.monotonic() - forecast_tick
        for k in FEATURES:
            require(tuple(forecast[k].shape[:2]) == (7, 5) and torch.isfinite(forecast[k]).all(), "Forecast layout/finiteness")
            expected_prefix = z_init[k].expand(5, *z_init[k].shape[1:]).transpose(0, 1)
            require(torch.equal(forecast[k][:1], expected_prefix), "Forecast context prefix misaligned")
            frozen["forecast_" + k] = forecast[k].numpy().copy()
        np.savez_compressed(directory / "forecast_commitment.npz", **frozen)
        write(directory / "forecast_commitment.json", {"sha256": digest(directory / "forecast_commitment.npz"),
              "candidate_outcomes_available": False, "bank_sha256": overall["bank_sha256"],
              "score_alpha": .1, "interpolation_lambdas": LAMBDAS})
        report["status"] = "replaying_committed_candidates"; persist()
        pixels, proprios = [], []
        baseline_target = root["target"].copy()
        for candidate in range(5):
            restore()
            cdir = directory / f"candidate{candidate}"; cdir.mkdir()
            rows, frames, props, integrations = [], [], [], []
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            with (cdir / "trace.jsonl").open("x") as handle:
                for step in range(30):
                    visual, info, row = trace_step(env, requests[candidate, step].clone(), 0, step)
                    report["activity"]["candidate_controls"] += 1
                    require(not env.proprio_env.unwrapped._did_see_sim_exception, "Native simulator exception")
                    require(not row["done"] and not row["truncated"], "Unexpected early termination")
                    require(np.array_equal(physical(env)["target"], baseline_target), "Physical task target changed")
                    row.update(candidate=candidate, seed=seed)
                    handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                    rows.append(row)
                    integ = np.empty(mujoco.mj_stateSize(env.proprio_env.unwrapped.model, spec))
                    mujoco.mj_getState(env.proprio_env.unwrapped.model, env.proprio_env.unwrapped.data, integ, spec)
                    integrations.append(integ)
                    if (step + 1) % 5 == 0:
                        frames.append(visual.numpy()[0].copy())
                        props.append(np.asarray(info["state"][:4], dtype=np.float32).copy())
            pixels.append(np.stack(frames)); proprios.append(np.stack(props))
            np.savez_compressed(cdir / "actual_future.npz", pixels=pixels[-1], proprio=proprios[-1],
                                full_integration_state=np.stack(integrations), rewards=[r["reward"] for r in rows])
            report["candidate_utilities"].append(utilities(rows)); persist()
        restore()
        env.close(); env = None
        report["status"] = "encoding_actual_futures"; persist()
        pixels, proprios = np.stack(pixels), np.stack(proprios)
        actual = [encode(pixels[:, h, None], proprios[:, h, None]) for h in range(6)]
        for k in FEATURES:
            frozen["truth_" + k] = torch.stack([z[k][:, 0] for z in actual]).numpy().copy()
            require(frozen["truth_" + k].shape[:2] == (6, 5) and np.isfinite(frozen["truth_" + k]).all(),
                    "Nonfinite or misaligned actual-future features")
            require(np.array_equal(z_init[k].numpy(), frozen["initial_" + k]), "Frozen context mutated")
            require(np.array_equal(z_goal[k].numpy(), frozen["goal_" + k]), "Frozen goal mutated")
        objective = ReprTargetDistMPCObjective(cfg, target_enc=z_goal, **cfg.planner.planning_objective)
        require(objective.alpha == .1 and objective.sum_all_diffs is False, "Native objective changed")
        scores, errors = tensor_analysis(frozen, torch, np, objective, actions)
        report["conditions"] = summarize_conditions(scores, errors, report["candidate_utilities"])
        report["horizon_scope"] = "All selections use terminal30-control goal score. h15 reports the prefix utility of those same H30-selected actions; it is not an H15 oracle policy."
        report["protected_checks"] = {"fresh_seed": seed not in (1, 2, 3, 4), "same_restored_root_each_candidate": True,
                                      "same_task_target_each_control": True, "context_goal_unchanged": True,
                                      "forecast_committed_before_candidate_replay": True,
                                      "native_score_exact_to_fp32_expression": True,
                                      "fixed_B5_reduction": True, "oracle_final_error_zero": True,
                                      "monotone_interpolation_fidelity": True, "cuda_initialized": torch.cuda.is_initialized()}
        require(not torch.cuda.is_initialized(), "CUDA initialized")
        np.savez_compressed(directory / "features.npz", **frozen)
        csv_rows = []
        for label, condition in report["conditions"].items():
            for candidate in range(5):
                row = {"seed": seed, "lambda": label, "candidate": candidate,
                       "native_cost": condition["native_costs"][candidate],
                       "selected": condition["selected_candidate"] == candidate}
                row.update({"forecast_error_" + k: errors[label][k][candidate] for k in (*FEATURES, "weighted")})
                row.update({h + "_" + k: v for h, values in report["candidate_utilities"][candidate].items() for k, v in values.items()})
                csv_rows.append(row)
        with (directory / "candidate_metrics.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
        report["artifact_sha256"] = {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob("*"))
                                     if p.is_file() and p.name != "result.json" and not p.name.endswith(".tmp")}
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({"seed": seed, "status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "original_selected": report["conditions"]["0.0"]["selected_candidate"],
                      "oracle_selected": report["conditions"]["1.0"]["selected_candidate"],
                      "terminal_utility_delta": report["conditions"]["0.0"]["metrics"]["h30"]["terminal_reward"]["Pred"]}), flush=True)
    return report


def main(args):
    protocol = json.loads(args.config.read_text())
    require(protocol["source_bank"]["count"] == 5, "Only frozen five-candidate protocol supported")
    require(protocol["runtime"]["unrolls_per_root"] == 1, "Require frozen oracle-only protocol, not alias protocol")
    require(protocol["oracle_feature_interpolation"]["lambdas"] == LAMBDAS, "Oracle intervention not pre-frozen")
    seeds = protocol["root_partition"]["engineering_calibration_physical_seeds" if args.phase == "calibration" else "confirmatory_physical_seeds"]
    require(seeds == ([1000] if args.phase == "calibration" else list(range(1001, 1031))), "Seed panel changed")
    bank_file = ROOT / protocol["source_bank"]["file"]
    require(digest(bank_file) == protocol["source_bank"]["sha256"], "Frozen candidate bank hash differs")
    out = args.out.resolve()
    if not out.exists():
        require(not args.verify_only, "No existing experiment to verify")
        out.mkdir(parents=True)
        shutil.copyfile(args.config, out / "launched_config.json")
        shutil.copyfile(__file__, out / "launched_script.py")
        shutil.copyfile(bank_file, out / "frozen_bank.npz")
    require(digest(out / "launched_config.json") == digest(args.config), "Cannot change protocol while resuming")
    require(digest(out / "launched_script.py") == digest(__file__), "Cannot change runner while resuming")
    require(digest(out / "frozen_bank.npz") == digest(bank_file), "Run-local bank changed")
    import numpy as np
    import torch
    torch.set_num_threads(protocol["runtime"]["threads"])
    completed, pending = [], []
    for seed in seeds:
        directory = out / f"seed{seed}"
        if directory.exists():
            completed.append(verify_root(directory, torch, np))
        else:
            pending.append(seed)
    if args.verify_only:
        require(not pending, "Missing registered roots: " + str(pending))
        print(json.dumps({"status": "verified", "roots": len(completed), "model_calls": 0, "environment_steps": 0})); return
    if not pending:
        print(json.dumps({"status": "already_complete_verified_no_rerun", "roots": len(completed)})); return
    overall = {"status": "running", "phase": args.phase, "seeds": seeds, "completed_seeds": [r["seed"] for r in completed],
               "scope": "Added fixed-bank open-loop30-control oracle-feature diagnostic, not native CEM or deployment defense.",
               "bank_sha256": digest(bank_file), "protocol_sha256": digest(args.config), "runner_sha256": digest(__file__),
               "interpolation_lambdas": LAMBDAS, "checkpoint_loads": {}, "network_attempts": 0,
               "training_updates": 0, "api_calls": 0, "model_load_calls": 0}
    overall["source_sha256"] = {name: digest(ROOT / name) for name in (
        "scripts/pilot_dino_metaworld_clean_gate.py", "scripts/probe_dino_action_saturation_gate.py",
        "vendor/jepa-wms/evals/simu_env_planning/planning/planning/objectives.py",
        "vendor/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py")}
    overall["control_count_scope"] = "Explicit expert100 + root warmup1 + candidate150 controls per root; any constructor-internal simulation is not instrumented."
    started = time.monotonic()
    def persist():
        overall["invocation_elapsed_seconds"] = time.monotonic() - started
        overall["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(out / "summary.json", overall)
    def timeout(*_):
        raise TimeoutError("Frozen process wall limit reached")
    def no_network(*_, **__):
        overall["network_attempts"] += 1
        raise RuntimeError("Network prohibited by CPU experiment protocol")
    wall = protocol["runtime"]["calibration_wall_limit_seconds" if args.phase == "calibration" else "panel_wall_limit_seconds"]
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, wall)
    persist()
    try:
        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=no_network))
            import mujoco
            from pilot_dino_metaworld_clean_gate import cfg_for, load_model, MODEL, DINO_WEIGHTS
            require(mujoco.__version__ == "3.3.0" and str(ROOT / ".deps/dino_mujoco330") in mujoco.__file__, "Wrong MuJoCo runtime")
            require(not torch.cuda.is_initialized(), "CUDA initialized")
            overall["runtime"] = {"interpreter": sys.executable, "torch": torch.__version__, "mujoco": mujoco.__version__,
                                  "mujoco_file": mujoco.__file__, "device": "cpu", "threads": torch.get_num_threads(), "renderer": "osmesa"}
            cfg = cfg_for(out, "cpu")
            original_load = torch.load
            def counted_load(path, *a, **kw):
                name = str(Path(path).resolve())
                require(name in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())), "Unexpected checkpoint")
                require(name not in overall["checkpoint_loads"], "Checkpoint loaded more than once")
                overall["checkpoint_loads"][name] = 1
                return original_load(path, *a, **kw)
            tick = time.monotonic()
            with patch.object(torch, "load", side_effect=counted_load):
                model, pre, overall["model_load"] = load_model(cfg, "cpu")
            require(not model.training and all(p.device.type == "cpu" and not p.requires_grad for p in model.parameters()),
                    "Model must remain frozen in CPU eval mode")
            overall["model_load_calls"] = 1
            overall["model_load_seconds"] = time.monotonic() - tick
            with np.load(out / "frozen_bank.npz", allow_pickle=False) as bank:
                norm = torch.from_numpy(bank["normalized"][0].copy())
                requests = torch.from_numpy(bank["requested_physical"][0].copy())
            require(tuple(norm.shape) == (5, 30, 4) and torch.equal(pre.denormalize_actions(norm), requests), "Native bank denormalization changed")
            persist()
            for seed in pending:
                result = execute_root(seed, out, cfg, model, pre, norm, requests, torch, np, mujoco, overall)
                verify_root(out / f"seed{seed}", torch, np)
                completed.append(result)
                overall["completed_seeds"].append(seed)
                require(len({r["root_goal_fingerprint"] for r in completed}) == len(completed), "Duplicated physical root+goal")
                persist()
            overall["status"] = "completed"
            overall["cuda_initialized"] = torch.cuda.is_initialized()
            require(not overall["cuda_initialized"], "CUDA initialized")
    except BaseException as error:
        overall.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        persist()
    print(json.dumps({"status": overall["status"], "roots": len(completed),
                      "elapsed_seconds": overall["invocation_elapsed_seconds"], "checkpoint_loads": len(overall["checkpoint_loads"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--phase", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/decision_leverage_dino_cpu_pilot")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        from probe_dino_action_saturation_gate import bootstrap
        bootstrap()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(args)
