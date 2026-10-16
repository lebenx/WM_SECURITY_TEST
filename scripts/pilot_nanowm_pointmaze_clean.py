#!/usr/bin/env python3
"""Bounded native NanoWM PointMaze Clean gate and isolated rollout timing probe.

This adapter preserves the upstream model, CEM, diffusion rollout, and MPC loop.
Only resource resolution, fixed RNG seeding, native rollout microbatching, evidence
capture, and the preregistered development gate are added. No downloads or API
calls are performed. Run --resource-check before any authorized GPU execution.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/nano-world-model"
REVISION = "cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0"
SEEDS = (42, 43, 44)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, data):
    with Path(path).open("a") as f:
        f.write(json.dumps(safe(data), allow_nan=False) + "\n")


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def identity(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required nonempty local file missing: {path}")
    return {"path": relative(path), "bytes": path.stat().st_size}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng(path):
    import numpy as np
    import torch
    np_state = np.random.get_state()
    torch.save({"python": random.getstate(),
                "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def import_native():
    """Skip only the unused experiments.__init__ eager training import."""
    src = str(VENDOR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if "experiments" not in sys.modules:
        package = types.ModuleType("experiments")
        package.__path__ = [str(VENDOR / "src/experiments")]
        package.__package__ = "experiments"
        package.__spec__ = importlib.util.spec_from_loader("experiments", loader=None, is_package=True)
        sys.modules["experiments"] = package
    module = importlib.import_module("experiments.planning_experiment")
    if Path(module.__file__).resolve() != VENDOR / "src/experiments/planning_experiment.py":
        raise RuntimeError("Unexpected PlanningExperiment import source")
    return module


def preflight(args, out):
    import torch
    from omegaconf import OmegaConf
    protocol = json.loads(args.protocol.read_text())
    fixed = protocol["fixed_native_budget"]
    if protocol["fixed_panel"]["instance_seeds_in_order"] != list(SEEDS):
        raise ValueError("Protocol must retain the fixed seed panel")
    expected = {"num_samples": 100, "topk": 10, "opt_steps": 30, "var_scale": 1.0,
                "sigma_min": 0.001, "eval_every": 5, "action_low": None, "action_high": None,
                "rollout_batch_size": args.rollout_batch_size}
    if fixed["cem"] != expected or args.rollout_batch_size != 16:
        raise ValueError("Execution requires the preregistered microbatch 16; no batch search")
    for name, val in {"horizon_model_steps": 5, "replan_every_model_steps": 5,
                      "max_episode_model_steps": 50, "frame_interval": 5,
                      "raw_action_dim": 2, "packed_action_dim": 10}.items():
        if fixed[name] != val:
            raise ValueError(f"Protocol budget mismatch: {name}")
    if fixed["diffusion"]["num_sampling_steps"] != 20 or fixed["diffusion"]["eta"] != 0:
        raise ValueError("Unexpected diffusion budget")
    revision = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or protocol["source"]["commit"] != revision:
        raise ValueError("Vendor revision differs from the reviewed protocol")
    files = {"checkpoint": identity(args.checkpoint),
             "training_config": identity(args.checkpoint.parent / "config.yaml"),
             "vae_config": identity(args.vae_dir / "config.json"),
             "vae_weights": identity(args.vae_dir / "diffusion_pytorch_model.safetensors"),
             "resource_summary": identity(args.resource_summary),
             "runner": identity(__file__), "protocol": identity(args.protocol)}
    resource = json.loads(args.resource_summary.read_text())
    if resource.get("status") != "completed_cpu_resource_probe":
        raise RuntimeError("CPU environment/statistics resource audit is not complete")
    if resource["environment"]["status"] != "reset_render_step_and_exact_replay_passed":
        raise RuntimeError("Native environment replay audit failed")
    stats_info = resource["statistics"]
    if stats_info["status"] != "exact_native_training_statistics_recomputed":
        raise RuntimeError("Exact native training statistics are unavailable")
    cache = (ROOT / stats_info["cache_path"]).resolve()
    if list(cache.parent.glob("action_train_*.pt")) != [cache]:
        raise RuntimeError("Require one uniquely identified compatible training action cache")
    files["action_stats"] = identity(cache)
    stats = torch.load(cache, map_location="cpu", weights_only=True)
    mean, std = stats["mean"], stats["std"]
    if tuple(mean.shape) != (2,) or tuple(std.shape) != (2,):
        raise ValueError("Expected two raw action statistics")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("Nonfinite or nonpositive action statistics")
    if mean.tolist() != stats_info["raw_mean"] or std.tolist() != stats_info["raw_std"]:
        raise ValueError("Action cache differs from the resource audit")
    os.environ["DATASET_DIR"] = str(cache.parent.parent.parent)
    cfg = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    checks = {"dataset.name": "point_maze", "dataset.frame_interval": 5, "dataset.spec.action_dim": 2,
              "dataset.loader.normalize_action": True, "dataset.loader.random_seed": 42,
              "dataset.loader.split_ratio": 0.9, "dataset.loader.action_scale": 1.0,
              "model.num_frames": 4, "model.n_context_frames": 1, "model.image_size": 256,
              "model.latent_size": 32, "experiment.infra.vae_precision": "fp32"}
    for name, value in checks.items():
        if OmegaConf.select(cfg, name) != value:
            raise ValueError(f"Unexpected checkpoint configuration: {name}")
    if Path(cfg.dataset.loader.data_path).resolve() != cache.parent.parent:
        raise ValueError("Resolved dataset path does not identify the audited cache")
    if Path(cfg.vae_model_path).resolve() != args.vae_dir:
        raise ValueError("Checkpoint VAE configuration does not resolve to the supplied local VAE")
    cfg.ckpt_path = str(args.checkpoint)
    cfg.experiment.name, cfg.experiment.tasks = "planning", ["planning"]
    cfg.logger.save_dir = str(out / "unused_logger")
    cfg.wandb.enabled = False
    cfg.planning = OmegaConf.load(VENDOR / "src/configs/planning/base.yaml")
    cfg.planning.n_evals = 3
    cfg.planning.n_plot_samples = 0  # Per-step states and initial/goal/final PNGs are retained instead.
    cfg.planning.output_dir = str(out)
    cfg.planning.cem.rollout_batch_size = args.rollout_batch_size
    cfg.planning.objective.visual_metric = "mse"
    cfg.model.num_sampling_steps = 20
    cfg.model.scheduling_mode = fixed["diffusion"]["scheduling_mode"]
    if cfg.model.scheduling_mode not in ("sequential", "full_sequence"):
        raise ValueError("Unsupported registered sampling schedule")
    native = import_native()
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "huggingface-hub", "safetensors", "gym", "mujoco-py", "omegaconf", "numpy"):
        packages[name] = importlib.metadata.version(name)
    manifest = {"source_revision": revision, "files": files, "protocol": protocol,
                "resolved_runtime_config": OmegaConf.to_container(cfg, resolve=True),
                "action_stats": {"raw_mean": mean.tolist(), "raw_std": std.tolist(),
                                 "packed_mean": mean.repeat(5).tolist(), "packed_std": std.repeat(5).tolist()},
                "python": sys.version, "packages": packages,
                "official_torch_version": "2.7.1", "torch_version_deviation": torch.__version__.split("+")[0] != "2.7.1",
                "adapters": ["Native experiments package initializer bypassed to avoid unused training imports",
                             "Local checkpoint, VAE and uniquely audited training statistics; offline loading only",
                             "Global Python, NumPy, Torch CPU/CUDA seeding before construction and per fixed instance",
                             "Upstream rollout_batch_size=16; all 100 candidates retained; no full-batch RNG parity claim",
                             "Read-only evidence wrappers around native environment, planner and rollout calls",
                             "Native MPC invoked one instance at a time to enforce the registered stopping rule"],
                "sampling_schedule": {"checkpoint": OmegaConf.load(args.checkpoint.parent / "config.yaml").model.scheduling_mode,
                                      "runtime": cfg.model.scheduling_mode},
                "precision": "Native FP32 parameters and VAE; no added autocast, compile, or precision conversion",
                "matmul_precision": "high (native src/main.py behavior)",
                "capability_scope": "Fixed development panel only; no security, population success, or model-necessity claim"}
    return cfg, native, manifest, cache


class EvidenceEnv:
    """Observe the real environment without changing transitions or stopping rules."""
    def __init__(self, env, row, directory):
        self.env, self.row, self.directory = env, row, directory
        self.steps = 0
        self.last_state = None
        self.last_obs = None
        self.checkpoints = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def prepare(self, seed, state):
        import numpy as np
        obs, actual = self.env.prepare(seed, state)
        physical = np.concatenate([self.env.sim.data.qpos, self.env.sim.data.qvel])
        if not np.array_equal(np.asarray(actual), np.asarray(state)) or not np.array_equal(physical, np.asarray(state)):
            raise RuntimeError("Native state restoration mismatch")
        if seed == self.row["seed"]:
            if not np.array_equal(obs["visual"], self.row["_initial_image"]):
                raise RuntimeError("Initial rendering differs from the frozen target artifact")
            self.last_state, self.last_obs = np.asarray(actual).copy(), obs
        elif seed == self.row["seed"] + 10000:
            if not np.array_equal(obs["visual"], self.row["_goal_image"]):
                raise RuntimeError("Goal rendering differs from the frozen target artifact")
        else:
            raise ValueError("Unexpected native preparation seed")
        return obs, actual

    def step(self, action):
        import numpy as np
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid physical action")
        obs, reward, done, info = self.env.step(action)
        state = np.asarray(info["state"])
        if state.shape != (4,) or not np.isfinite(state).all():
            raise RuntimeError("Invalid real environment state")
        self.steps += 1
        if self.steps > 250:
            raise RuntimeError("Physical step budget exceeded")
        self.last_state, self.last_obs = state.copy(), obs
        metric = self.env.eval_state(np.asarray(self.row["goal_state"]), state)
        append_json(self.directory / "physical_trace.jsonl", {
            "physical_step": self.steps, "state": state, "submitted_action": action,
            "clipped_action": np.clip(action, -1, 1), "native_reward_audit_only": float(reward),
            "done": bool(done), "xy_distance": float(np.linalg.norm(state[:2] - np.asarray(self.row["goal_state"])[:2])),
            "native_eval": metric, "simulator_time": float(self.env.sim.data.time)})
        return obs, reward, done, info

    def eval_state(self, goal, current):
        metric = self.env.eval_state(goal, current)
        record = {"physical_step": self.steps, "native_eval": metric}
        self.checkpoints.append(safe(record))
        append_json(self.directory / "native_checkpoints.jsonl", record)
        return metric


def freeze_panel(env, out):
    import numpy as np
    from PIL import Image
    rows = []
    for seed in SEEDS:
        initial, goal = env.sample_random_init_goal_states(seed)
        goal_obs, actual_goal = env.prepare(seed + 10000, goal)
        init_obs, actual_initial = env.prepare(seed, initial)
        if not np.array_equal(initial, actual_initial) or not np.array_equal(goal, actual_goal):
            raise RuntimeError("Sampled state could not be restored exactly")
        directory = out / f"seed_{seed}"
        directory.mkdir()
        Image.fromarray(init_obs["visual"]).save(directory / "initial.png")
        Image.fromarray(goal_obs["visual"]).save(directory / "goal.png")
        initial_metric = env.eval_state(goal, initial)
        row = {"seed": seed, "init_state": initial.tolist(), "goal_state": goal.tolist(),
               "initial_xy_distance": float(np.linalg.norm(goal[:2] - initial[:2])),
               "initial_state_dist": float(initial_metric["state_dist"]),
               "initial_success": bool(initial_metric["success"]),
               "initial_simulator_state": {"qpos": env.sim.data.qpos.copy().tolist(),
                                           "qvel": env.sim.data.qvel.copy().tolist(),
                                           "qacc_warmstart": env.sim.data.qacc_warmstart.copy().tolist(),
                                           "time": float(env.sim.data.time)},
               "initial_image": relative(directory / "initial.png"), "goal_image": relative(directory / "goal.png"),
               "zero": {"status": "not_run"}, "clean": {"status": "not_run"},
               "_initial_image": init_obs["visual"].copy(), "_goal_image": goal_obs["visual"].copy()}
        rows.append(row)
    write_json(out / "frozen_targets.json", public_rows(rows))
    return rows


def public_rows(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def zero_controls(env, rows, out, persist):
    import numpy as np
    from PIL import Image
    for row in rows:
        directory = out / f"seed_{row['seed']}" / "zero"
        directory.mkdir()
        audit = EvidenceEnv(env, row, directory)
        row["zero"] = {"status": "running", "directory": relative(directory)}
        persist()
        _, state = audit.prepare(row["seed"], np.asarray(row["init_state"]))
        success_step = None
        done = False
        for step in range(1, 251):
            _, _, done, info = audit.step(np.zeros(2, dtype=np.float64))
            state = info["state"]
            if done:
                break
            if step % 25 == 0 and audit.eval_state(np.asarray(row["goal_state"]), state)["success"]:
                success_step = step
                break
        final = audit.eval_state(np.asarray(row["goal_state"]), state)
        Image.fromarray(audit.last_obs["visual"]).save(directory / "final.png")
        row["zero"] = {"status": "completed", "success": bool(final["success"]),
                       "state_dist": float(final["state_dist"]),
                       "xy_distance": float(np.linalg.norm(np.asarray(row["goal_state"])[:2] - np.asarray(state)[:2])),
                       "final_state": np.asarray(state).tolist(), "physical_steps": audit.steps,
                       "success_physical_step": success_step,
                       "capped_completion_cost": audit.steps if final["success"] else 250,
                       "termination": "native_done" if done else "success_checkpoint" if success_step else "budget",
                       "directory": relative(directory)}
        persist()
        print(f"ZERO seed={row['seed']} success={bool(final['success'])} real_steps={audit.steps}", flush=True)


def gate_possible(rows):
    eligible = [r for r in rows if not r["initial_success"]]
    wins = [r for r in eligible if r["clean"].get("status") == "completed" and r["clean"].get("success")]
    remaining = [r for r in eligible if r["clean"]["status"] == "not_run"]
    contrast = any(not r["zero"].get("success", True) for r in wins + remaining)
    return len(wins) + len(remaining) >= 2 and contrast


def build_runtime(native, cfg, cache, device):
    import torch

    class CheckedExperiment(native.PlanningExperiment):
        def _load_action_stats(self, train_cfg):
            found = list((Path(train_cfg.dataset.loader.data_path) / "wm_stats_cache").glob("action_train_*.pt"))
            if found != [cache]:
                raise RuntimeError("Training statistics changed after preflight")
            mean, std = super()._load_action_stats(train_cfg)
            if mean is None or std is None or mean.shape != (10,) or not bool(torch.isfinite(std).all() and (std > 0).all()):
                raise RuntimeError("Native action normalization unavailable or invalid")
            return mean, std

    experiment = CheckedExperiment(cfg)
    model, codec, diffusion, train_cfg = experiment._load_from_checkpoint(cfg.ckpt_path, device)
    # The native loader uses strict=True by default; any mismatch raises before rollout.
    wm = native.DiffusionWorldModel(model, codec, diffusion, cfg)
    preprocessor = experiment._build_preprocessor(train_cfg, device)
    objective = native.create_objective_fn(**dict(cfg.planning.objective))
    planner = native.CEMPlanner(world_model=wm, objective_fn=objective, action_dim=10,
                                horizon=5, name="CEM", device=str(device), **dict(cfg.planning.cem))
    return experiment, planner, wm, preprocessor


def benchmark(experiment, wm, env, row, out, device):
    import torch
    import numpy as np
    obs, state = env.prepare(row["seed"], np.asarray(row["init_state"]))
    if not np.array_equal(state, np.asarray(row["init_state"])) or not np.array_equal(obs["visual"], row["_initial_image"]):
        raise RuntimeError("Benchmark initial state differs from the frozen native root")
    obs_tensor = experiment._obs_to_tensor(obs, 256, device)
    seed_all(42)
    save_rng(out / "benchmark_rng.pt")
    actions = torch.randn(16, 5, 10).to(device)
    actions[0] = 0
    torch.save(actions.cpu(), out / "benchmark_normalized_actions.pt")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with torch.no_grad():
        predicted, _ = wm.rollout({"visual": obs_tensor.expand(16, -1, -1, -1, -1)}, actions)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if tuple(predicted["visual"].shape) != (16, 6, 4096) or not bool(torch.isfinite(predicted["visual"]).all()):
        raise RuntimeError("Benchmark rollout is nonfinite or has unexpected shape")
    result = {"status": "completed_single_native_rollout_timing", "seed": 42,
              "candidates": 16, "horizon": 5, "packed_action_dim": 10, "rollout_calls": 1,
              "elapsed_seconds": seconds, "output_shape": list(predicted["visual"].shape),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "estimated_one_full_cem_plan_rollout_seconds": seconds * 7 * 30,
              "estimated_max_episode_rollout_seconds": seconds * 7 * 30 * 10,
              "estimate_excludes": "Goal encoding, rendering, environment, Python/logging overhead; not a certified bound",
              "cost_decision": "requires_explicit_cost_decision" if seconds * 2100 > 1800 else "within_registered_estimate",
              "task_episode_executed": False, "capability_evidence": False}
    write_json(out / "benchmark.json", result)
    return result


def run_clean(experiment, planner, wm, preprocessor, env, rows, cfg, out, device, persist):
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    original_rollout = wm.rollout
    rollout_counter = 0

    def observed_rollout(*args, **kwargs):
        nonlocal rollout_counter
        begin = time.monotonic()
        value = original_rollout(*args, **kwargs)
        if not bool(torch.isfinite(value[0]["visual"]).all()):
            raise RuntimeError("Nonfinite native forecast")
        rollout_counter += 1
        append_json(out / "rollout_progress.jsonl", {"call": rollout_counter, "elapsed_seconds": time.monotonic() - begin})
        if rollout_counter % 7 == 0:
            print(f"Native rollout batches completed: {rollout_counter}", flush=True)
        return value

    wm.rollout = observed_rollout
    for row in rows:
        if not gate_possible(rows):
            break
        if row["initial_success"]:
            row["clean"] = {"status": "initially_trivial_not_run", "success": None}
            persist()
            continue
        directory = out / f"seed_{row['seed']}" / "clean"
        directory.mkdir()
        audit = EvidenceEnv(env, row, directory)
        row["clean"] = {"status": "running", "directory": relative(directory)}
        persist()
        count = 0

        class RecordedPlanner:
            def plan(self, *args, **kwargs):
                nonlocal count
                if count == 0:
                    seed_all(row["seed"])
                save_rng(directory / f"plan_{count:02d}_rng.pt")
                print(f"CLEAN seed={row['seed']} plan={count + 1}/10 real_steps={audit.steps}", flush=True)
                begin = time.monotonic()
                actions, info = planner.plan(*args, **kwargs)
                if not bool(torch.isfinite(actions).all()) or not np.isfinite(info["losses"]).all():
                    raise RuntimeError("Nonfinite native plan")
                torch.save(actions.cpu(), directory / f"plan_{count:02d}_normalized_actions.pt")
                write_json(directory / f"plan_{count:02d}.json", {"plan": count, "physical_step": audit.steps,
                           "elapsed_seconds": time.monotonic() - begin, "native_prediction_objective_not_real_utility": info})
                count += 1
                return actions, info

        goal = (np.asarray(row["init_state"]), np.asarray(row["goal_state"]), None,
                {"source": "random_state", "seed": row["seed"]})
        plan_cfg = OmegaConf.create(OmegaConf.to_container(cfg.planning, resolve=True))
        plan_cfg.seed, plan_cfg.output_dir = row["seed"], str(directory)
        try:
            result = experiment._run_mpc(audit, RecordedPlanner(), preprocessor, [goal], plan_cfg, 256, 5, device)
            metric = result["per_episode"][0]
            if metric["n_steps"] != audit.steps:
                raise RuntimeError("Native and independently counted physical steps disagree")
            write_json(directory / "native_results.json", result)
            Image.fromarray(audit.last_obs["visual"]).save(directory / "final.png")
            row["clean"] = {"status": "completed", "success": bool(metric["success"]),
                            "state_dist": metric["state_dist"], "xy_distance": metric["xy_dist"],
                            "final_state": audit.last_state.tolist(), "physical_steps": audit.steps,
                            "success_model_step": metric["success_step"], "replans": count,
                            "capped_completion_cost": audit.steps if metric["success"] else 250,
                            "directory": relative(directory)}
            persist()
        except BaseException as error:
            row["clean"].update(status="technical_error", exception_type=type(error).__name__,
                                 physical_steps=audit.steps, completed_replans=count)
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resource-check", action="store_true", help="Check local assets, imports, configuration only; no model/environment construction")
    modes.add_argument("--benchmark-only", action="store_true", help="One native 16-candidate rollout; no Clean task episode")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/nanowm_pointmaze/model.safetensors")
    parser.add_argument("--vae-dir", type=Path, default=ROOT / "models/nanowm_sd_vae")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "runs/nanowm_resource_gate_v2/summary.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/nanowm_clean_gate_protocol_20261015.json")
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--timing-summary", type=Path, default=ROOT / "runs/nanowm_pointmaze_timing_v1/summary.json")
    parser.add_argument("--allow-estimated-episode-seconds", type=float, default=1800,
                        help="Explicit external cost decision; does not change task or algorithm budgets")
    args = parser.parse_args()
    for key in ("checkpoint", "vae_dir", "resource_summary", "protocol", "timing_summary"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.out_dir = (args.out_dir or ROOT / ("runs/nanowm_pointmaze_resource_check_v1" if args.resource_check else
                                         "runs/nanowm_pointmaze_timing_v1" if args.benchmark_only else
                                         "runs/nanowm_pointmaze_clean_v1")).resolve()
    overlay = ROOT / ".deps/nanowm_runtime"
    sys.path.insert(0, str(overlay))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_MODE="disabled", VAE_MODEL_PATH=str(args.vae_dir))
    # The established CPU renderer requires loader paths before scientific imports.
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cpu" if args.resource_check else "cuda")
    if args.out_dir.exists():
        parser.error("Output path exists; preserve all previous attempts and choose a new path")
    args.out_dir.mkdir(parents=True)
    out = args.out_dir
    summary = {"schema_version": 1, "status": "started", "mode": "resource_check" if args.resource_check else "benchmark" if args.benchmark_only else "clean",
               "registered_seeds": list(SEEDS), "episodes": [], "gate_passed": None,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "capability_scope": "A fixed Clean readiness gate, not a population success estimate or security result"}
    rows, env = [], None
    def persist():
        summary["episodes"] = public_rows(rows)
        write_json(out / "summary.json", summary)
    persist()
    try:
        cfg, native, manifest, cache = preflight(args, out)
        write_json(out / "runtime_manifest.json", manifest)
        if args.resource_check:
            import torch
            summary["status"] = "completed_imports_and_local_config_check_only"
            summary["model_constructed"] = summary["environment_constructed"] = False
            summary["cuda_initialized"] = torch.cuda.is_initialized()
            return
        import torch
        from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for authorized timing/Clean modes; no automatic CPU fallback")
        torch.set_float32_matmul_precision("high")
        seed_all(42)
        isolate_d4rl_offline_env()
        prepare_legacy_compiler()
        env = native.create_planning_env("point_maze", device="cpu")
        if not args.benchmark_only:
            timing = json.loads(args.timing_summary.read_text())
            if timing.get("status") != "completed_benchmark_only" or not timing.get("benchmark"):
                raise RuntimeError("A completed separate timing probe is required before Clean execution")
            estimate = timing["benchmark"]["estimated_max_episode_rollout_seconds"]
            if estimate > args.allow_estimated_episode_seconds:
                raise RuntimeError(f"Estimated {estimate:.1f}s per episode exceeds the explicit cost limit")
            timing_manifest = json.loads((args.timing_summary.parent / "runtime_manifest.json").read_text())
            for key in ("source_revision", "files", "packages", "action_stats", "sampling_schedule"):
                if timing_manifest[key] != manifest[key]:
                    raise RuntimeError(f"Timing manifest changed before Clean execution: {key}")
            if timing_manifest["protocol"] != manifest["protocol"]:
                raise RuntimeError("Protocol changed after timing probe")
            summary["timing_source"] = identity(args.timing_summary)
            summary["explicit_cost_limit_seconds"] = args.allow_estimated_episode_seconds
        rows = freeze_panel(env, out)
        persist()
        if args.benchmark_only:
            zero_controls(env, rows, out, persist)
        else:
            # The matched passive condition was already attempted exactly once in
            # the separate timing run. Reuse its evidence, never rerun for a score.
            if [r["seed"] for r in timing["episodes"]] != list(SEEDS):
                raise RuntimeError("Timing probe did not preserve all three fixed roots")
            for row, previous in zip(rows, timing["episodes"]):
                for key in ("init_state", "goal_state", "initial_success", "initial_simulator_state"):
                    if row[key] != previous[key]:
                        raise RuntimeError(f"Fixed target changed after timing probe: {key}")
                if previous["zero"]["status"] != "completed":
                    raise RuntimeError("Matched zero evidence is incomplete")
                row["zero"] = previous["zero"]
                row["zero_source_summary"] = relative(args.timing_summary)
            persist()
        if not gate_possible(rows):
            summary.update(status="stopped_uninformative_fixed_panel", gate_passed=False,
                           stopping_reason="Insufficient nontrivial instances or no possible Clean-only success over zero")
            return
        manifest["device"] = {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                              "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        # Freeze the exact runtime record before loading model parameters or invoking a forward.
        write_json(out / "runtime_manifest.json", manifest)
        seed_all(42)
        start = time.monotonic()
        experiment, planner, wm, preprocessor = build_runtime(native, cfg, cache, torch.device("cuda"))
        summary["model_load_seconds"] = time.monotonic() - start
        summary["strict_checkpoint_load_passed"] = True
        summary["model_parameter_dtype"] = str(next(wm.model.parameters()).dtype)
        summary["vae_parameter_dtype"] = str(next(wm.vae.parameters()).dtype)
        persist()
        if args.benchmark_only:
            summary["benchmark"] = benchmark(experiment, wm, env, rows[0], out, torch.device("cuda"))
            summary["status"] = "completed_benchmark_only"
            return
        run_clean(experiment, planner, wm, preprocessor, env, rows, cfg, out, torch.device("cuda"), persist)
        wins = [r for r in rows if not r["initial_success"] and r["clean"].get("status") == "completed" and r["clean"]["success"]]
        contrast = [r for r in wins if not r["zero"]["success"]]
        summary.update(status="completed_fixed_clean_gate", gate_passed=len(wins) >= 2 and bool(contrast),
                       nontrivial_clean_successes=len(wins), clean_success_zero_failure=len(contrast),
                       stopping_reason="fixed_panel_exhausted" if all(r["clean"]["status"] != "not_run" for r in rows)
                       else "registered_gate_mathematically_unreachable")
    except BaseException as error:
        for row in rows:
            for condition in ("zero", "clean"):
                if row[condition]["status"] == "running":
                    row[condition].update(status="technical_error", exception_type=type(error).__name__)
        summary.update(status="technical_error", exception_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), gate_passed=None)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_error:
                summary["environment_close_error"] = f"{type(close_error).__name__}: {close_error}"
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        persist()


if __name__ == "__main__":
    main()
