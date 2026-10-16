#!/usr/bin/env python3
"""Native NanoWM Clean gate on three frozen feasible heldout PointMaze tasks.

No reference actions or source trajectories are loaded. --resource-check performs
CPU-only imports/configuration checks and never starts the shared-GPU wait.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import pilot_nanowm_pointmaze_clean as base

ROOT = base.ROOT
PROTOCOL = ROOT / "reports/nanowm_heldout_clean_protocol_20261015.json"
TASKS = ROOT / "runs/nanowm_heldout_pointmaze_tasks_v1"
TARGET_KEYS = {"instance", "seed", "trajectory_id", "offset", "source_split", "init_state", "goal_state", "initial_image", "goal_image"}


class ResourceDeferred(Exception):
    """An incomplete resource-limited attempt; never a completed Clean failure."""


def smi(query):
    text = subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [[item.strip() for item in row] for row in csv.reader(text.splitlines()) if row]


def gpu_sample(index):
    inventory = smi("--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu")
    matches = [row for row in inventory if int(row[0]) == index]
    if len(matches) != 1:
        raise RuntimeError("Selected physical GPU is unavailable")
    gpu = matches[0]
    processes = smi("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    other = [{"pid": int(row[1]), "process_name": row[2], "memory_mib": row[3]}
             for row in processes if row[0] == gpu[1] and int(row[1]) != os.getpid()]
    return {"gpu_index": index, "uuid": gpu[1], "name": gpu[2], "memory_total_mib": gpu[3],
            "memory_used_mib": gpu[4], "utilization_percent": gpu[5], "other_compute_processes": other}


def wait_for_gpu(index, out, summary, persist, deadline_utc=None):
    start = time.monotonic()
    wait_budget = 1800.0
    if deadline_utc is not None:
        wait_budget = min(wait_budget, max(0.0, deadline_utc.timestamp() - time.time()))
    summary["gpu_wait_policy"] = {
        "maximum_wait_seconds": 1800,
        "this_attempt_wait_budget_seconds": wait_budget,
        "supplied_deadline_utc": None if deadline_utc is None else deadline_utc.isoformat(),
        "effective_deadline_utc": datetime.fromtimestamp(time.time() + wait_budget, timezone.utc).isoformat(),
        "idle_first_sample_may_proceed": True,
    }
    while True:
        sample = gpu_sample(index)
        elapsed = time.monotonic() - start
        sample.update(wait_seconds=elapsed, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        base.append_json(out / "gpu_wait.jsonl", sample)
        summary.update(status="waiting_for_shared_gpu", gpu_wait=sample)
        persist()
        if not sample["other_compute_processes"]:
            return sample
        if elapsed >= wait_budget:
            raise ResourceDeferred("Shared GPU remained occupied until the bounded wait deadline")
        print(f"Waiting for GPU {index}: {len(sample['other_compute_processes'])} other compute PIDs; {elapsed:.0f}/{wait_budget:.0f} s", flush=True)
        time.sleep(min(30, wait_budget - elapsed))


def load_targets(protocol):
    import numpy as np
    from PIL import Image
    targets = json.loads((TASKS / "planner_targets.json").read_text())
    # Only status, instance identifiers, and zero metrics are extracted from this
    # audit summary. Its reference paths are never opened or passed to a planner.
    audit_summary = json.loads((TASKS / "summary.json").read_text())
    if audit_summary.get("status") != "completed_cpu_task_selection_gate" or not audit_summary.get("task_selection_gate_passed"):
        raise RuntimeError("CPU task-selection gate is incomplete")
    zeros = {item["instance"]: item["zero"] for item in audit_summary["instances"]}
    del audit_summary
    if len(targets) != 3:
        raise ValueError("Require the exact three fixed targets")
    rows, artifacts = [], []
    for index, target in enumerate(targets):
        if set(target) != TARGET_KEYS:
            raise ValueError("Unexpected target field; reference or source data is not allowed")
        expected = {"instance": index, "seed": 42 + index, "trajectory_id": protocol["fixed_tasks"]["trajectory_ids"][index],
                    "offset": protocol["fixed_tasks"]["offsets"][index], "source_split": "validation"}
        if any(target[key] != value for key, value in expected.items()):
            raise ValueError("Frozen task order or identity differs from protocol")
        initial, goal = (np.asarray(target[key], dtype=np.float32) for key in ("init_state", "goal_state"))
        if initial.shape != (4,) or goal.shape != (4,) or not np.isfinite([initial, goal]).all():
            raise ValueError("Invalid frozen task state")
        row = dict(target)
        for field, private, filename in (("initial_image", "_initial_image", "initial.png"), ("goal_image", "_goal_image", "goal.png")):
            path = (ROOT / target[field]).resolve()
            if path != (TASKS / f"instance_{index:03d}" / filename).resolve():
                raise ValueError("Image path is outside the fixed input allowlist")
            artifacts.append(base.identity(path))
            with Image.open(path) as image:
                row[private] = np.asarray(image).copy()
            if row[private].shape != (224, 224, 3) or row[private].dtype != np.uint8:
                raise ValueError("Expected native RGB uint8 observations")
        zero = zeros[index]
        if zero["status"] != "completed" or zero["success"] != protocol["fixed_tasks"]["zero_success"][index]:
            raise RuntimeError("Matched zero condition is incomplete or changed")
        if zero["physical_steps"] > 50:
            raise RuntimeError("Zero control uses an incompatible budget")
        row.update(initial_xy_distance=float(np.linalg.norm(goal[:2] - initial[:2])),
                   initial_state_dist=float(np.linalg.norm(goal - initial)),
                   initial_success=bool(np.linalg.norm(goal[:2] - initial[:2]) < 0.5),
                   zero=zero, clean={"status": "not_run"})
        if row["initial_success"]:
            raise RuntimeError("This frozen panel was registered with three nontrivial roots")
        rows.append(row)
    return rows, artifacts


def prepare_config(args, out, protocol):
    from omegaconf import OmegaConf
    legacy = SimpleNamespace(checkpoint=ROOT / "models/nanowm_pointmaze/model.safetensors",
                             vae_dir=ROOT / "models/nanowm_sd_vae", resource_summary=ROOT / "runs/nanowm_resource_gate_v2/summary.json",
                             protocol=ROOT / "reports/nanowm_clean_gate_protocol_20261015.json", rollout_batch_size=16)
    cfg, native, manifest, cache = base.preflight(legacy, out)
    budget = protocol["budget"]
    for key, value in {"horizon": 5, "replan_every": 5, "max_episode_steps": 10, "frame_interval": 5,
                       "max_replans": 2, "max_physical_steps": 50}.items():
        if budget[key] != value:
            raise ValueError(f"Unexpected heldout budget: {key}")
    if budget["cem"] != dict(cfg.planning.cem) or budget["diffusion"] != {"num_sampling_steps": 20, "scheduling_mode": "full_sequence", "eta": 0.0}:
        raise ValueError("Unexpected CEM or diffusion configuration")
    cfg.planning.max_episode_steps = 10
    cfg.planning.goal_source = "dset"
    manifest["resource_preflight_only"] = {"helper": manifest["files"]["runner"], "legacy_protocol": manifest["files"]["protocol"],
                                           "scope": "Resource/import/statistics checks only; its max50 task configuration is not executed"}
    manifest["files"]["runner"] = base.identity(__file__)
    manifest["files"]["protocol"] = base.identity(PROTOCOL)
    manifest["files"]["planner_targets"] = base.identity(TASKS / "planner_targets.json")
    manifest["files"]["task_summary_zero_only"] = base.identity(TASKS / "summary.json")
    manifest["protocol"] = protocol
    manifest["resolved_runtime_config"] = OmegaConf.to_container(cfg, resolve=True)
    manifest["task_adapter"] = "Frozen validation PlanningGoal inputs; no native source dataset/action loader is invoked"
    return cfg, native, manifest, cache


def cost_probe(experiment, wm, env, row, directory, device, resource_check):
    import numpy as np
    import torch
    directory.mkdir()
    observation, state = env.prepare(row["seed"], np.asarray(row["init_state"], dtype=np.float32))
    if not np.array_equal(state, row["init_state"]) or not np.array_equal(observation["visual"], row["_initial_image"]):
        raise RuntimeError("Probe reset differs from frozen task observation")
    tensor = experiment._obs_to_tensor({"visual": row["_initial_image"]}, 256, device)
    base.seed_all(42)
    base.save_rng(directory / "rng.pt")
    actions = torch.randn(16, 5, 10).to(device)
    actions[0] = 0
    torch.save(actions.cpu(), directory / "normalized_actions.pt")
    resource_check("before_cost_probe")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        prediction, _ = wm.rollout({"visual": tensor.expand(16, -1, -1, -1, -1)}, actions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {"status": "completed_single_native_rollout", "instance": 0, "seed": 42, "candidates": 16,
              "rollout_calls": 1, "elapsed_seconds": elapsed, "output_shape": list(prediction["visual"].shape),
              "finite": bool(torch.isfinite(prediction["visual"]).all()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "estimated_two_plan_rollout_seconds": elapsed * 7 * 30 * 2,
              "estimate_excludes": "Goal encoding, rendering, environment, logging and control overhead; not a certified bound",
              "clean_episode": False, "task_success": None, "reference_actions_loaded": False}
    base.write_json(directory / "result.json", result)
    if not result["finite"] or result["output_shape"] != [16, 6, 4096]:
        raise RuntimeError("Cost probe output is nonfinite or incompatible")
    resource_check("after_cost_probe")
    return result


def clean_episode(experiment, planner, native, preprocessor, env, row, cfg, directory, device, resource_check):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    directory.mkdir()

    class AuditEnv(base.EvidenceEnv):
        def step(self, action):
            if self.steps >= 50:
                raise RuntimeError("Registered 50-physical-step budget exceeded")
            return super().step(action)

    audit = AuditEnv(env, row, directory)
    count = 0

    class RecordedPlanner:
        def plan(self, *args, **kwargs):
            nonlocal count
            resource_check("before_plan")
            if count == 0:
                base.seed_all(row["seed"])
            if count >= 2:
                raise RuntimeError("Registered two-plan budget exceeded")
            base.save_rng(directory / f"plan_{count:02d}_rng.pt")
            print(f"CLEAN instance={row['instance']} seed={row['seed']} plan={count + 1}/2 physical_steps={audit.steps}", flush=True)
            start = time.monotonic()
            actions, info = planner.plan(*args, **kwargs)
            if not bool(torch.isfinite(actions).all()) or not np.isfinite(info["losses"]).all():
                raise RuntimeError("Nonfinite native plan")
            torch.save(actions.cpu(), directory / f"plan_{count:02d}_normalized_actions.pt")
            base.write_json(directory / f"plan_{count:02d}.json", {"plan": count, "elapsed_seconds": time.monotonic() - start,
                            "physical_step": audit.steps, "native_prediction_loss_not_real_utility": info})
            count += 1
            resource_check("after_plan_before_real_actions")
            return actions, info

    goal = native.PlanningGoal(init_state=np.asarray(row["init_state"], dtype=np.float32),
                               goal_state=np.asarray(row["goal_state"], dtype=np.float32), env_info=None,
                               goal_meta={key: row[key] for key in ("instance", "seed", "trajectory_id", "offset", "source_split")},
                               init_obs={"visual": row["_initial_image"]}, goal_obs={"visual": row["_goal_image"]})
    plan_cfg = OmegaConf.create(OmegaConf.to_container(cfg.planning, resolve=True))
    plan_cfg.seed, plan_cfg.output_dir = row["seed"], str(directory)
    try:
        result = experiment._run_mpc(audit, RecordedPlanner(), preprocessor, [goal], plan_cfg, 256, 5, device)
        metric = result["per_episode"][0]
        if metric["n_steps"] != audit.steps or audit.steps > 50:
            raise RuntimeError("Independent physical-step count mismatch")
        base.write_json(directory / "native_results.json", result)
        Image.fromarray(audit.last_obs["visual"]).save(directory / "final.png")
        return {"status": "completed", "success": bool(metric["success"]), "state_dist": metric["state_dist"],
                "xy_distance": metric["xy_dist"], "final_state": audit.last_state.tolist(), "physical_steps": audit.steps,
                "success_model_step": metric["success_step"], "replans": count,
                "capped_completion_cost": audit.steps if metric["success"] else 50, "directory": base.relative(directory)}
    except BaseException:
        row["clean"].update(physical_steps=audit.steps, completed_replans=count,
                            partial_state=None if audit.last_state is None else audit.last_state.tolist())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-check", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--gpu-wait-deadline-utc", help="Optional timezone-aware ISO8601 cutoff; can only shorten the registered wait")
    args = parser.parse_args()
    deadline = None
    if args.gpu_wait_deadline_utc:
        try:
            deadline = datetime.fromisoformat(args.gpu_wait_deadline_utc.replace("Z", "+00:00"))
            if deadline.utcoffset() is None:
                raise ValueError("timezone is required")
            deadline = deadline.astimezone(timezone.utc)
        except ValueError as error:
            parser.error(f"Invalid GPU wait deadline: {error}")
    out = (args.out_dir or ROOT / ("runs/nanowm_heldout_clean_resource_check_v1" if args.resource_check else "runs/nanowm_heldout_clean_v1")).resolve()
    sys.path.insert(0, str(ROOT / ".deps/nanowm_runtime"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                      VAE_MODEL_PATH=str(ROOT / "models/nanowm_sd_vae"))
    # This only prepares the legacy CPU renderer's loader paths and restarts once.
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cpu" if args.resource_check else "cuda")
    if out.exists():
        parser.error("Output exists; previous attempts must be retained")
    out.mkdir(parents=True)
    rows, env, task_start = [], None, None
    summary = {"schema_version": 1, "status": "started", "clean_gate_passed": None, "instances": [],
               "reference_actions_loaded": False, "scope": "Fixed heldout Clean capability gate, not a population estimate or security result",
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def persist():
        summary["instances"] = base.public_rows(rows)
        base.write_json(out / "summary.json", summary)
    persist()
    try:
        protocol = json.loads(PROTOCOL.read_text())
        cfg, native, manifest, cache = prepare_config(args, out, protocol)
        rows, manifest["input_images"] = load_targets(protocol)
        base.write_json(out / "runtime_manifest.json", manifest)
        base.write_json(out / "planner_targets.json", [{key: row[key] for key in TARGET_KEYS} for row in rows])
        persist()
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before the shared-GPU gate")
        if args.resource_check:
            summary.update(status="completed_cpu_import_config_and_target_check", cuda_initialized=False,
                           model_constructed=False, environment_constructed=False, gpu_wait_started=False)
            return
        gpu = wait_for_gpu(protocol["shared_gpu"]["gpu_index"], out, summary, persist, deadline)
        # Legacy mujoco_py parses this value as an integer during rendering.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["gpu_index"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before selecting the idle GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("Selected idle GPU is unavailable to Torch")
        torch.set_float32_matmul_precision("high")
        base.seed_all(42)
        from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler
        isolate_d4rl_offline_env()
        prepare_legacy_compiler()
        env = native.create_planning_env("point_maze", device="cpu")
        manifest["device"] = {"selected_uuid": gpu["uuid"], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "name": torch.cuda.get_device_name(),
                              "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        base.write_json(out / "runtime_manifest.json", manifest)

        def resource_check(stage):
            elapsed = None if task_start is None else time.monotonic() - task_start
            if elapsed is not None and elapsed >= 8100:
                raise ResourceDeferred("Clean task phase exceeded its registered 135-minute wall-clock limit")
            sample = gpu_sample(protocol["shared_gpu"]["gpu_index"])
            sample.update(stage=stage, task_elapsed_seconds=elapsed)
            base.append_json(out / "gpu_execution.jsonl", sample)
            if sample["uuid"] != gpu["uuid"] or sample["other_compute_processes"]:
                raise ResourceDeferred("Another compute process appeared on the selected GPU; partial task retained")

        resource_check("before_model_load")
        summary["status"] = "loading_model"
        persist()
        start = time.monotonic()
        experiment, planner, wm, preprocessor = base.build_runtime(native, cfg, cache, torch.device("cuda"))
        summary.update(model_load_seconds=time.monotonic() - start, strict_checkpoint_load_passed=True,
                       model_parameter_dtype=str(next(wm.model.parameters()).dtype), vae_parameter_dtype=str(next(wm.vae.parameters()).dtype))
        if summary["model_parameter_dtype"] != "torch.float32" or summary["vae_parameter_dtype"] != "torch.float32":
            raise RuntimeError("Native FP32 parameter requirement was not met")
        persist()
        summary["cost_probe"] = cost_probe(experiment, wm, env, rows[0], out / "cost_probe", torch.device("cuda"), resource_check)
        persist()
        if summary["cost_probe"]["estimated_two_plan_rollout_seconds"] > 2700:
            raise ResourceDeferred("One-shot rollout estimate exceeds the registered 45-minute per-root cost decision")
        native_rollout = wm.rollout
        rollout_count = 0
        def observed_rollout(*args, **kwargs):
            nonlocal rollout_count
            resource_check("before_native_rollout")
            value = native_rollout(*args, **kwargs)
            if not bool(torch.isfinite(value[0]["visual"]).all()):
                raise RuntimeError("Nonfinite native prediction")
            rollout_count += 1
            summary["completed_clean_microbatch_rollouts"] = rollout_count
            if rollout_count % 7 == 0:
                persist()
                print(f"Completed native Clean rollout batches: {rollout_count}", flush=True)
            return value
        wm.rollout = observed_rollout
        task_start = time.monotonic()
        summary.update(status="running_fixed_clean_panel", task_started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        for row in rows:
            if not base.gate_possible(rows):
                break
            resource_check("before_episode")
            row["clean"] = {"status": "running"}
            persist()
            row["clean"] = clean_episode(experiment, planner, native, preprocessor, env, row, cfg,
                                          out / f"instance_{row['instance']:03d}", torch.device("cuda"), resource_check)
            persist()
        wins = [r for r in rows if r["clean"].get("status") == "completed" and r["clean"].get("success")]
        contrast = [r for r in wins if not r["zero"]["success"]]
        summary.update(status="completed_fixed_clean_gate", clean_gate_passed=len(wins) >= 2 and bool(contrast),
                       clean_success_count=len(wins), clean_success_zero_failure=len(contrast),
                       stopping_reason="fixed_panel_exhausted" if all(r["clean"]["status"] != "not_run" for r in rows)
                       else "registered_gate_mathematically_unreachable")
    except ResourceDeferred as error:
        for row in rows:
            if row["clean"]["status"] == "running":
                row["clean"].update(status="resource_deferred", success=None)
        summary.update(status="resource_deferred", reason=str(error), clean_gate_passed=None)
    except BaseException as error:
        for row in rows:
            if row["clean"]["status"] == "running":
                row["clean"].update(status="technical_error", success=None)
        summary.update(status="technical_error", exception_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), clean_gate_passed=None)
        raise
    finally:
        if task_start is not None:
            summary["task_elapsed_seconds"] = time.monotonic() - task_start
        if env is not None:
            try:
                env.close()
            except Exception as error:
                summary["environment_close_error"] = str(error)
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        persist()


if __name__ == "__main__":
    main()
