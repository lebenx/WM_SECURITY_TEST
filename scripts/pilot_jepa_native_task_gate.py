#!/usr/bin/env python3
"""One released JEPA-WM PointMaze planning episode, CPU by default.

Uses the upstream encoder/predictor, CEM, objective, physics and pixel wrappers.
The released H6/nas6/random-state protocol plans ONCE then executes 30 controls
(150 MuJoCo substeps); this is not a receding-horizon feedback experiment.
Only optional image/state decoding/plots are disabled. CPU device and RNG
initialization are explicit portability adaptations; no CUDA parity is claimed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/jepa-wms"
CONFIG = UPSTREAM / "configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
CHECKPOINT = ROOT / "models/JEPA_WM/mz_jepa-wm.pth.tar"
EXPECTED_SHA = "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc"
DINOV2 = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DINO_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
SDK = ROOT / "vendor/runtime/mujoco210/mujoco210"
OSMESA = ROOT / ".deps/osmesa/usr"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def configure_process(device):
    """Restart before scientific imports so the ELF loader sees local libraries."""
    lib = str(OSMESA / "lib/x86_64-linux-gnu")
    bins = str(SDK / "bin")
    env = {
        "MUJOCO_PY_FORCE_CPU": "1",
        "MUJOCO_PY_MUJOCO_PATH": str(SDK), "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1",
        "LD_LIBRARY_PATH": lib + ":" + bins + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        "LIBRARY_PATH": lib + ":" + os.environ.get("LIBRARY_PATH", ""),
        "CPATH": str(OSMESA / "include") + ":" + os.environ.get("CPATH", ""),
        "PATH": str(Path(sys.executable).parent) + ":" + os.environ.get("PATH", ""),
    }
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if os.environ.get("JEPA_NATIVE_CPU_BOOTSTRAPPED") != "1":
        os.environ.update(env)
        os.environ["JEPA_NATIVE_CPU_BOOTSTRAPPED"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def isolate_d4rl_offline_env():
    """Load its real OfflineEnv without registering unrelated benchmark suites."""
    spec = importlib.util.find_spec("d4rl")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("D4RL missing in the isolated environment")
    package = types.ModuleType("d4rl")
    package.__path__ = list(spec.submodule_search_locations)
    package.__spec__ = spec
    sys.modules["d4rl"] = package
    # Only upstream offline_env and infos are required by this MazeEnv.
    from d4rl import offline_env
    return Path(offline_env.__file__)


def prepare_legacy_compiler():
    """uv Python names GNU cc as cc; legacy distutils then emits invalid -R.

    Use the same installed GNU compiler under its recognized gcc name so
    distutils forwards the library rpath via -Wl. Only compilation is affected.
    """
    from distutils import sysconfig
    values = sysconfig.get_config_vars()
    if values.get("CC", "").split()[0] == "cc":
        values["CC"] = values["CC"].replace("cc", "gcc", 1)


def load_model(cfg, device="cpu"):
    import torch
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    mk = cfg["model_kwargs"]
    stats = get_data_stats("pointmaze")
    pre = Preprocessor(
        **{key: torch.tensor(stats[key]) for key in (
            "action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")},
        transform=make_transforms(img_size=224, normalize=mk["data_aug"]["normalize"],
                                  random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                  random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False),
    )
    pretrain = copy.deepcopy(mk["pretrain_kwargs"])
    pretrain["heads_cfg"] = {}  # Optional visualization readouts; objective uses predicted features.
    hub_load = torch.hub.load
    state_load = torch.nn.Module.load_state_dict
    checks = []

    def local_hub(repo, name, *args, **kwargs):
        if repo != "facebookresearch/dinov2" or name != "dinov2_vits14":
            raise ValueError(f"Unexpected remote model load: {repo}/{name}")
        kwargs["weights"] = str(DINO_WEIGHTS)
        return hub_load(str(DINOV2), name, *args, source="local", **kwargs)

    def checked_state_load(module, state, *args, **kwargs):
        result = state_load(module, state, *args, **kwargs)
        record = {"class": type(module).__name__, "missing": list(result.missing_keys),
                  "unexpected": list(result.unexpected_keys)}
        checks.append(record)
        if record["missing"] or record["unexpected"]:
            raise RuntimeError(f"Checkpoint mismatch: {record}")
        return result

    with patch.object(torch.hub, "load", side_effect=local_hub), patch.object(torch.nn.Module, "load_state_dict", checked_state_load):
        model = init_module(folder=CHECKPOINT.parent, checkpoint=CHECKPOINT.name,
                            model_kwargs=pretrain, device=torch.device(device),
                            action_dim=stats["action_dim"], proprio_dim=stats["proprio_dim"],
                            preprocessor=pre, cfgs_data=mk["data"], wrapper_kwargs=mk["wrapper_kwargs"])
    model.eval().requires_grad_(False)
    return model, pre, stats, checks


def build_agent(cfg, model, pre):
    import torch
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    if cfg.device == "cuda":
        return GC_Agent(cfg, model, dset=None, preprocessor=pre)
    original = torch.Generator

    def cpu_generator(*args, **kwargs):
        if str(kwargs.get("device", "cpu")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original(*args, **kwargs)

    # Upstream algorithm methods remain intact; only CUDA initializer redirected.
    with patch.object(torch, "Generator", side_effect=cpu_generator):
        agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
    agent.device = torch.device("cpu")
    agent.planner.device = torch.device("cpu")
    return agent


def physical_state(env):
    maze = env.unwrapped
    data = maze.sim.data
    return {"time": float(data.time), "qpos": data.qpos.copy().tolist(),
            "qvel": data.qvel.copy().tolist(), "qacc_warmstart": data.qacc_warmstart.copy().tolist(),
            "ctrl": data.ctrl.copy().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/jepa_native_task_gate_v0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-memory-limit-gib", type=float, default=6.,
                        help="Allocator ceiling; keep native batch size/precision and preserve OOM evidence")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--model-only", action="store_true", help="Only inspect/load CPU model; never report task success")
    args = parser.parse_args()
    configure_process(args.device)
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty; preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {"status": "preparing", "episodes": [], "api_requests": 0, "training_updates": 0,
               "device": args.device, "renderer": "mujoco-py LinuxCPUExtensionBuilder / OSMesa",
               "gpu_used": args.device == "cuda", "script_sha256": sha(__file__)}
    write_json(out / "summary.json", summary)
    env = None
    try:
        import numpy as np
        import torch
        import yaml
        from omegaconf import OmegaConf
        from PIL import Image
        from tensordict import TensorDict
        sys.path.insert(0, str(UPSTREAM))
        torch.set_num_threads(args.threads)
        if args.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(args.cuda_memory_limit_gib * 2**30 / total, 1.))
            torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True  # Native eval.py setting.
        logging.getLogger().setLevel(logging.WARNING)
        original_cfg = yaml.safe_load(CONFIG.read_text())
        cfg = OmegaConf.create(original_cfg)
        cfg.work_dir = out
        cfg.device = args.device
        cfg.rank, cfg.world_size, cfg.local_seed = 0, 1, int(cfg.meta.seed)
        cfg.frameskip = int(cfg.model_kwargs.data.custom.frameskip)
        cfg.action_ratio = 5
        cfg.tasks = [cfg.task_specification.task]
        cfg.task_specification.multitask = False
        cfg.task_specification.max_episode_steps = cfg.frameskip * cfg.task_specification.goal_H
        cfg.logging.optional_plots = False
        cfg.planner.decode_each_iteration = False
        cfg.action_dim = 2
        if (cfg.planner.iterations, cfg.planner.num_samples, cfg.planner.num_elites,
                cfg.planner.horizon, cfg.planner.num_act_stepped) != (30, 300, 10, 6, 6):
            raise RuntimeError("Native planner protocol changed")
        random.seed(cfg.meta.seed)
        np.random.seed(cfg.meta.seed)
        torch.manual_seed(cfg.meta.seed)
        if sha(CHECKPOINT) != EXPECTED_SHA:
            raise RuntimeError("PointMaze checkpoint differs from pinned artifact")
        sources = [CONFIG, Path(__file__), UPSTREAM / "hubconf.py"]
        sources += list((UPSTREAM / "evals/simu_env_planning").rglob("*.py"))
        sources += list((UPSTREAM / "app/vjepa_wm").rglob("*.py"))
        sources += list((UPSTREAM / "app/plan_common/models").rglob("*.py"))
        sources += list((UPSTREAM / "src/models").rglob("*.py"))
        sources += [UPSTREAM / "app/plan_common/datasets" / f for f in ("__init__.py", "preprocessor.py", "transforms.py")]
        protocol = {
            "upstream_config": str(CONFIG), "original_config": original_cfg,
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_SHA,
            "encoder_checkpoint": str(DINO_WEIGHTS), "encoder_sha256": sha(DINO_WEIGHTS),
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in (
                "torch", "torchvision", "timm", "tensordict", "numpy", "gym", "mujoco-py", "d4rl", "nevergrad")},
            "device": args.device, "threads": args.threads,
            "cuda_allocator_limit_gib": args.cuda_memory_limit_gib if args.device == "cuda" else None,
            "encoder_source_sha256": {str(p.relative_to(DINOV2)): sha(p) for p in DINOV2.rglob("*.py")},
            "physics_library_sha256": sha(SDK / "bin/libmujoco210.so"),
            "osmesa_library_sha256": sha(OSMESA / "lib/x86_64-linux-gnu/libOSMesa.so.8"),
            "episode_seed_rule": "(local_seed**2 + episode_index*local_seed) % (2**32-2)",
            "episode_index": args.episode_index, "local_seed": int(cfg.local_seed),
            "planner_unchanged": "Upstream CEM.plan, goal objective, encoder and autoregressive unroll; H6/300 candidates/30 iterations/10 elites/alpha0.1/context2.",
            "temporal_contract": "Each model action has10 entries = five2D controls, not one repeated control;6 macro actions execute30 controls, each5 MuJoCo steps. Default has one planning call, no feedback replan.",
            "success_definition": "Upstream eval_state: final XY Euclidean distance to independently sampled physical goal state <0.5; native done_at_succ=False.",
            "adaptations": ["Separate CPU baseline; native GC_Agent CUDA generators and device fields redirected to CPU." if args.device == "cpu" else "Native CUDA model/CEM generators and FP32 precision; software OSMesa renderer runs on CPU.",
                            "Use upstream hub DATA_STATS and exact validation transform, no training dataset needed for random_state goals.",
                            "Optional decoder heads/plots disabled; they do not participate in CEM objective or latent feedback.",
                            "Load cached DINOv2 source/checkpoint locally, no remote fallback.",
                            "Only D4RL OfflineEnv/infos imported; unrelated benchmark auto-registration skipped.",
                            "Legacy distutils compilation recognizes local GNU cc as gcc to emit valid linker rpath; no simulator source changed.",
                            "Python3.11/torch2.6 compatibility environment differs from upstream package pins; no CUDA numerical parity claimed."],
            "scope": "One preregistered first native random-state episode; success is a runtime gate, not a success-rate estimate or attack evidence.",
        }
        write_json(out / "protocol.json", protocol)
        model, pre, stats, load_checks = load_model(original_cfg, args.device)
        # Native main_distributed_episodes_eval reseeds after model construction.
        random.seed(cfg.meta.seed)
        np.random.seed(cfg.meta.seed)
        torch.manual_seed(cfg.meta.seed)
        summary.update(model_loaded=True, model_parameter_count=sum(p.numel() for p in model.parameters()),
                       checkpoint_load_checks=load_checks, normalization_stats=stats, model_action_dimension=model.action_dim)
        write_json(out / "summary.json", summary)
        if args.model_only:
            summary.update(status="model_loaded_only", cuda_initialized=torch.cuda.is_initialized())
            write_json(out / "summary.json", summary)
            return
        offline_source = isolate_d4rl_offline_env()
        prepare_legacy_compiler()
        from evals.simu_env_planning.envs.pointmaze_gym_wrap import make_env
        from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
        from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
        env = PixelWrapper(cfg, TensorWrapper(make_env(cfg)))
        agent = build_agent(cfg, model, pre)
        ep_seed = (int(cfg.local_seed)**2 + args.episode_index * int(cfg.local_seed)) % (2**32 - 2)
        initial, goal = env.sample_random_init_goal_states(ep_seed)
        goal_obs, goal_info = env.prepare(ep_seed, goal)
        goal_td = TensorDict({"visual": goal_obs, "proprio": goal_info["proprio"]})
        obs, info = env.prepare(ep_seed, initial)
        initial_state = physical_state(env)
        for name, frame in (("initial", obs), ("goal", goal_obs)):
            Image.fromarray(frame[-1].permute(1, 2, 0).numpy()).save(out / f"{name}.png")
        np.savez(out / "initial_goal.npz", sampled_initial=initial, sampled_goal=goal,
                 initial_state=np.asarray(info["state"]), goal_state=np.asarray(goal_info["state"]),
                 initial_pixels=obs.numpy(), goal_pixels=goal_obs.numpy())
        summary.update(status="planning", episode_seed=ep_seed, sampled_initial=initial.tolist(),
                       sampled_goal=goal.tolist(), initial_simulator_state=initial_state,
                       success_at_start=bool(env.eval_state(goal_info["state"], info["state"])["success"]),
                       offline_env_sha256=sha(offline_source), planning_calls=0, environment_steps=0,
                       native_reward_target=env.unwrapped.get_target().tolist(),
                       reward_scope="Native sparse reward target can be OFF_TARGET=(10,10) because wrapper uses with_target=False; task success uses sampled goal XY, never this reward as success proxy.")
        write_json(out / "summary.json", summary)
        agent.set_goal(goal_td)
        original_cost = agent.planner.cost_function
        iteration_rows = []

        def measured_cost(actions, z_init):
            if args.device == "cuda":
                torch.cuda.synchronize()
            tick = time.monotonic()
            cost = original_cost(actions, z_init)
            if args.device == "cuda":
                torch.cuda.synchronize()
            row = {"iteration": len(iteration_rows)+1, "candidate_count": int(actions.shape[1]),
                   "elapsed_seconds": time.monotonic()-tick, "cost_min": float(cost.min()),
                   "cost_max": float(cost.max()), "cost_mean": float(cost.mean()),
                   "all_finite": bool(torch.isfinite(cost).all())}
            if not row["all_finite"]:
                raise RuntimeError("Native CEM returned nonfinite costs")
            iteration_rows.append(row)
            write_json(out / "iterations.json", iteration_rows)
            print(json.dumps(row), flush=True)
            return cost

        agent.planner.cost_function = measured_cost
        observations, states, actions_all, rewards_all = [obs.clone()], [np.asarray(info["state"]).copy()], [], []
        done = False
        trace = []
        while not done:
            td = TensorDict({"visual": obs, "proprio": info["proprio"]})
            steps_left = max((env.steps_left()+1)*model.action_skip//cfg.frameskip, 1)
            t_plan = time.monotonic()
            normalized = agent.act(td, steps_left=steps_left)
            real_actions = pre.denormalize_actions(normalized.reshape(-1, 2))
            summary["planning_calls"] += 1
            torch.save({"normalized_macro_actions": normalized, "denormalized_controls": real_actions,
                        "losses": agent._prev_losses, "elapsed_seconds": time.monotonic()-t_plan},
                       out / f"plan_{summary['planning_calls']:02d}.pt")
            for action in real_actions:
                before = physical_state(env)
                obs, reward, done, truncated, info = env.step(action)
                success = bool(env.eval_state(goal_info["state"], info["state"])["success"])
                row = {"step": len(trace)+1, "action": action.tolist(), "physics_action_clipped": action.clamp(-1,1).tolist(),
                       "reward": float(reward), "done": bool(done), "truncated": bool(truncated),
                       "sampled_goal_success": success, "state": np.asarray(info["state"]).tolist(),
                       "xy_goal_distance": float(np.linalg.norm(np.asarray(info["state"])[:2]-goal_info["state"][:2])),
                       "before": before, "after": physical_state(env)}
                trace.append(row)
                with (out / "trace.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False)+"\n")
                observations.append(obs.clone()); states.append(np.asarray(info["state"]).copy())
                actions_all.append(action.clone()); rewards_all.append(float(reward))
                summary["environment_steps"] = len(trace)
                if done:
                    break
            write_json(out / "summary.json", summary)
        np.savez(out / "trajectory.npz", observations=torch.stack(observations).numpy(),
                 states=np.stack(states), actions=torch.stack(actions_all).numpy(), rewards=np.asarray(rewards_all))
        Image.fromarray(obs[-1].permute(1, 2, 0).numpy()).save(out / "final.png")
        assert len(trace) == 30 and summary["planning_calls"] == 1
        if args.device == "cpu":
            assert not torch.cuda.is_initialized()
        summary.update(status="completed", success=trace[-1]["sampled_goal_success"],
                       final_xy_goal_distance=trace[-1]["xy_goal_distance"], native_environment_return=sum(rewards_all),
                       actual_mujoco_substeps=5*len(trace), cuda_initialized=torch.cuda.is_initialized(),
                       final_simulator_state=physical_state(env),
                       core_gate="Single actual task result; no statistical clean-capability claim, and native protocol has no feedback replanning.",
                       episodes=[{"episode_index": args.episode_index, "seed": ep_seed,
                                  "success": trace[-1]["sampled_goal_success"], "environment_steps": len(trace),
                                  "native_return": sum(rewards_all), "planning_calls": 1}])
    except BaseException as exc:
        summary.update(status="resource_or_runtime_error", error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc(), task_failure_inferred=False)
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic()-started
        summary["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if "torch" in locals() and args.device == "cuda" and torch.cuda.is_initialized():
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(out / "summary.json", summary)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
