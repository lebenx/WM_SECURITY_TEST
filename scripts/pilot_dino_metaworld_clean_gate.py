#!/usr/bin/env python3
"""Fixed DINO-WM MetaWorld reach health gate, using the released native CEM.

CPU validation loads weights but never runs the model, then records matched
physical zero-control rollouts. The separately registered GPU run has three
Clean rollouts. Goal features enter only the native planning objective.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/jepa-wms"
OFFICIAL = UPSTREAM / "configs/evals/simu_env_planning/mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml"
MODEL = ROOT / "models/DINO_WM/mw_dino-wm.pth.tar"
DINO_SOURCE = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DINO_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
OSMESA = ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu"
DEFAULT_CONFIG = ROOT / "configs/dino_metaworld_clean_gate_v1.json"
DEFAULT_CPU = ROOT / "runs/dino_metaworld_clean_gate_cpu_v1"
DEFAULT_OUT = ROOT / "runs/dino_metaworld_clean_gate_v1"
SEEDS = [1, 2, 3]
GATE = {"all_initial_states_nontrivial": True, "all_expert_final_success": True,
        "all_exact_paired_initial_goal_and_noop_repeats": True,
        "minimum_clean_final_successes": 2, "minimum_clean_minus_noop_final_successes": 2,
        "complete_fixed_three_seeds_required": True,
        "interpretation": "Small development runtime/normal-task gate, not statistical confirmation, attack success or a forecast-only causal attribution",
        "stop_rule": "Preserve every fixed seed and failure. No replacement, longer horizon, more samples/iterations, retries, or seed search."}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(data)
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(data)
        temp.replace(path)


def bootstrap(cpu):
    """Software rendering for both branches; no shared dependency installation."""
    marker = "DINO_MW_CLEAN_BOOTSTRAPPED"
    if os.environ.get(marker) != "1":
        os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                          LIBGL_ALWAYS_SOFTWARE="1", **{marker: "1"})
        os.environ["LD_LIBRARY_PATH"] = str(OSMESA) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Software rendering backend changed")
    sys.path.insert(0, str(UPSTREAM))


def versions():
    names = ("torch", "torchvision", "numpy", "tensordict", "timm", "gym", "gymnasium", "mujoco", "metaworld", "Pillow", "clusterscope")
    return {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in names}}


def cfg_for(out, device):
    import yaml
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(yaml.safe_load(OFFICIAL.read_text()))
    require((cfg.planner.iterations, cfg.planner.num_samples, cfg.planner.num_elites,
             cfg.planner.horizon, cfg.planner.num_act_stepped) == (15, 300, 10, 6, 3), "Official CEM changed")
    require(cfg.task_specification.max_episode_steps == 100 and cfg.task_specification.goal_source == "expert"
            and cfg.task_specification.succ_def == "simu" and not cfg.task_specification.done_at_succ, "Official task protocol changed")
    require(cfg.model_kwargs.data.custom.frameskip == 5 and cfg.model_kwargs.data.custom.action_skip == 1
            and cfg.model_kwargs.wrapper_kwargs.ctxt_window == 2
            and cfg.planner.planning_objective.alpha == .1, "Official temporal/objective protocol changed")
    require(list(cfg.model_kwargs.data_aug.normalize[0]) == list(cfg.model_kwargs.data_aug.normalize[1]) == [.5, .5, .5], "Official RGB normalization changed")
    cfg.work_dir = str(out)
    cfg.device = device
    cfg.rank, cfg.world_size, cfg.local_seed = 0, 1, 1
    cfg.frameskip, cfg.action_ratio, cfg.action_dim = 5, 5, 4
    cfg.tasks = ["mw-reach"]
    cfg.task_specification.multitask = False
    cfg.logging.optional_plots = False
    cfg.logging.tqdm_silent = True
    cfg.planner.decode_each_iteration = False
    cfg.planner.distribute_planner = False
    cfg.model_kwargs.pretrain_kwargs.heads_cfg = {}
    return cfg


def load_model(cfg, device):
    import torch
    from omegaconf import OmegaConf
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    mk = OmegaConf.to_container(cfg.model_kwargs, resolve=False)
    stats = get_data_stats("metaworld")
    pre = Preprocessor(**{k: torch.tensor(stats[k]) for k in
        ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")},
        transform=make_transforms(img_size=224, normalize=mk["data_aug"]["normalize"],
            random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
            random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False))
    original_hub, original_load = torch.hub.load, torch.nn.Module.load_state_dict
    checks = []
    def hub(repo, name, *args, **kwargs):
        require(repo == "facebookresearch/dinov2" and name == "dinov2_vits14", "Unexpected model download")
        kwargs["weights"] = str(DINO_WEIGHTS)
        return original_hub(str(DINO_SOURCE), name, *args, source="local", **kwargs)
    def checked(module, state, *args, **kwargs):
        # Enforce exact learned/buffer schema even where upstream uses strict=False.
        result = original_load(module, state, strict=True)
        checks.append({"class": type(module).__name__, "loaded_tensors": len(state),
                       "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)})
        return result
    with patch.object(torch.hub, "load", side_effect=hub), patch.object(torch.nn.Module, "load_state_dict", checked):
        model = init_module(folder=MODEL.parent, checkpoint=MODEL.name, model_kwargs=mk["pretrain_kwargs"],
            device=torch.device(device), action_dim=stats["action_dim"], proprio_dim=stats["proprio_dim"],
            preprocessor=pre, cfgs_data=mk["data"], wrapper_kwargs=mk["wrapper_kwargs"])
    model.eval().requires_grad_(False)
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Model parameters are not FP32")
    require(model.action_dim == 20 and model.action_skip == 1 and model.tubelet_size_enc == 1, "Model action/time dimensions changed")
    report = {"strict_loads": checks, "parameters": sum(p.numel() for p in model.parameters()),
              "all_parameters_float32": True, "model_action_dimension": model.action_dim,
              "model_forward_calls": 0, "normalization": stats}
    return model, pre, report


def array_digest(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()}


def physical(env):
    import numpy as np
    raw = env.proprio_env.unwrapped
    fields = {"sim_" + k: np.asarray(getattr(raw.data, k)).copy() for k in
              ("qpos", "qvel", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")}
    fields.update(sim_time=np.asarray(raw.data.time), target=np.asarray(raw._target_pos).copy(),
                  rand_vec=np.asarray(raw._last_rand_vec).copy(), tcp=np.asarray(raw.tcp_center).copy(),
                  previous_observation=np.asarray(raw._prev_obs).copy(),
                  elapsed=np.asarray(env.elapsed_steps()), model_site_pos=raw.model.site_pos.copy())
    return fields


def setup_episode(cfg, seed, directory):
    """Replay native expert-goal setup; never use MW.prepare as state restoration."""
    import numpy as np
    import torch
    from PIL import Image
    from evals.simu_env_planning.envs.init import make_env
    # evals.utils eagerly imports all remote/training datasets solely to define
    # make_datasets, which expert goals never call. Fail if that unused path is
    # invoked, rather than installing network clients or loading a dataset.
    dataset_stub = types.ModuleType("app.plan_common.datasets.utils")
    def forbidden_dataset(*args, **kwargs):
        raise RuntimeError("Dataset initialization is outside this fixed expert-goal gate")
    dataset_stub.init_data = forbidden_dataset
    dataset_name = "app.plan_common.datasets.utils"
    previous_dataset_module = sys.modules.get(dataset_name)
    sys.modules[dataset_name] = dataset_stub
    try:
        from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
        import evals.simu_env_planning.planning.plan_evaluator as module
    finally:
        # Restore only our one import entry; preserve legitimately imported
        # native modules. Restoring all sys.modules would trigger unsafe reloads.
        if previous_dataset_module is None:
            sys.modules.pop(dataset_name, None)
        else:
            sys.modules[dataset_name] = previous_dataset_module
    # A fresh environment per paired branch is an explicit dev-gate adaptation.
    # CEM's separate CUDA Generator is unaffected by these global seed resets.
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    env = make_env(cfg)
    evaluator = PlanEvaluator(cfg, None)
    try:
        env.reset(seed=seed, task_idx=-1)
        raw = env.proprio_env.unwrapped
        raw._freeze_rand_vec = False
        raw.seeded_rand_vec = True
        env.seed(seed)
        # Capture/close the expert environment that native set_episode creates.
        experts, expert_trace = [], []
        def track_env(*args, **kwargs):
            created = make_env(*args, **kwargs)
            experts.append(created)
            original_expert_step = created.step
            def observe_expert(action, *step_args, **step_kwargs):
                result = original_expert_step(action, *step_args, **step_kwargs)
                expert_trace.append({"step": created.elapsed_steps(), "action": action.detach().cpu().tolist(),
                    "reward": float(result[1]), "done": bool(result[2]), "truncated": bool(result[3]),
                    "success": bool(result[4]["success"]), "state": np.asarray(result[4]["state"]).tolist()})
                return result
            created.step = observe_expert
            return created
        warmup = []
        original_step = env.step
        def capture_warmup(action, *args, **kwargs):
            result = original_step(action, *args, **kwargs)
            warmup.append({"state": np.asarray(result[4]["state"]).copy(),
                           "reward": float(result[1]), "success": bool(result[4]["success"])})
            return result
        try:
            env.step = capture_warmup
            with patch.object(module, "make_env", side_effect=track_env):
                initial, goal, expert_observations, expert_success = evaluator.set_episode(cfg, None, env, seed)
            expert = experts[0]
            require(len(experts) == 1 and expert.elapsed_steps() == 100, "Unexpected expert budget")
            expert_state = physical(expert)
            expert_actions = evaluator.expert_actions.detach().cpu().numpy()
        finally:
            env.step = original_step
            for e in experts:
                e.close()
        require(env.elapsed_steps() == 1 and len(warmup) == 1, "Native one-step warmup not preserved")
        arrays = physical(env)
        arrays.update(initial_pixels=initial["visual"].cpu().numpy(), initial_proprio=initial["proprio"].cpu().numpy(),
                      initial_state=warmup[0]["state"], goal_pixels=goal["visual"].cpu().numpy(),
                      goal_proprio=goal["proprio"].cpu().numpy(), goal_state=np.asarray(evaluator.state_g).copy(),
                      expert_actions=expert_actions,
                      expert_final_tcp=expert_state["tcp"], expert_target=expert_state["target"])
        distance = float(np.linalg.norm(arrays["tcp"] - arrays["target"]))
        expert_distance = float(np.linalg.norm(expert_state["tcp"] - expert_state["target"]))
        require(np.array_equal(arrays["target"], expert_state["target"])
                and np.array_equal(arrays["rand_vec"], expert_state["rand_vec"]), "Expert targets differ from task")
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "setup.npz", **arrays)
        require(len(expert_trace) == 100 and expert_trace[-1]["success"] == bool(expert_success), "Expert recorded trace/budget differs")
        with (directory / "expert_trace.jsonl").open("x") as handle:
            for event in expert_trace:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
        for name, td in (("initial", initial), ("goal", goal)):
            Image.fromarray(td["visual"][-1].permute(1, 2, 0).cpu().numpy()).save(directory / (name + ".png"))
        report = {"seed": seed, "episode_index": seed - 1, "elapsed_after_warmup": 1,
                  "initial_success": distance <= .05, "initial_reach_distance": distance,
                  "warmup_reward_excluded_from_policy_return": warmup[0]["reward"],
                  "warmup_native_success": warmup[0]["success"],
                  "expert_final_success": bool(expert_success), "expert_final_reach_distance": expert_distance,
                  "expert_steps": 100, "goal_source": "native separate-env expert final image/proprio; expert controls are archived but never provided to CEM",
                  "expert_trace": bind(directory / "expert_trace.jsonl"),
                  "arrays": {k: array_digest(v) for k, v in arrays.items()},
                  "setup": bind(directory / "setup.npz"), "initial_image": bind(directory / "initial.png"), "goal_image": bind(directory / "goal.png")}
        write(directory / "setup.json", report)
        return env, initial, goal, report
    except BaseException:
        env.close()
        raise


def trace_step(env, action, planning_call, local_index):
    import numpy as np
    before = physical(env)
    obs, reward, done, truncated, info = env.step(action)
    after = physical(env)
    row = {"environment_step": env.elapsed_steps(), "policy_step": env.elapsed_steps() - 1,
           "planning_call": planning_call, "action_in_chunk": local_index,
           "action": action.detach().cpu().numpy().tolist(), "reward": float(reward),
           "done": bool(done), "truncated": bool(truncated), "success": bool(info["success"]),
           "reach_distance": float(np.linalg.norm(after["tcp"] - after["target"])),
           "tcp_before": before["tcp"].tolist(), "tcp_after": after["tcp"].tolist(),
           "state": np.asarray(info["state"]).tolist(),
           "physical": {k: array_digest(v) for k, v in after.items()},
           "pixels": array_digest(obs.cpu().numpy())}
    return obs, info, row


def noop(env, directory):
    import torch
    rows = []
    with (directory / "trace.jsonl").open("x") as handle:
        while env.steps_left() > 0:
            _, info, row = trace_step(env, torch.zeros(4), 0, 0)
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if row["done"]: break
    return outcome(rows, directory)


def outcome(rows, directory):
    require(bool(rows), "No executed policy controls")
    result = {"final_success": rows[-1]["success"], "ever_success": any(r["success"] for r in rows),
              "final_reach_distance": rows[-1]["reach_distance"], "total_environment_steps": rows[-1]["environment_step"],
              "policy_controls_executed": len(rows), "reward_sum": sum(r["reward"] for r in rows),
              "terminal_done": rows[-1]["done"], "trace": bind(directory / "trace.jsonl")}
    require(result["total_environment_steps"] == 100 and result["policy_controls_executed"] == 99, "Native fixed budget changed")
    write(directory / "outcome.json", result)
    return result


def cpu_check(out):
    import torch
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    cfg = cfg_for(out, "cpu")
    report = {"status": "running", "runner": bind(__file__), "versions": versions(), "seeds": SEEDS,
              "episodes": [], "model_forward_calls": 0, "GPU_used": False, "api_requests": 0, "training_updates": 0}
    write(out / "summary.json", report)
    try:
        model, _, loading = load_model(cfg, "cpu")
        report["checkpoint"] = bind(MODEL)
        report["model_load"] = loading
        require(not torch.cuda.is_initialized(), "CPU validation initialized CUDA")
        del model
        for seed in SEEDS:
            episode = {"seed": seed, "status": "running", "repeats": []}
            report["episodes"].append(episode)
            try:
                for repeat in range(2):
                    directory = out / f"seed{seed}_repeat{repeat}"
                    env, initial, goal, setup = setup_episode(cfg, seed, directory)
                    try:
                        result = noop(env, directory)
                    finally:
                        env.close()
                    episode["repeats"].append({"setup": setup, "noop": result})
                left, right = episode["repeats"]
                checks = {"all_setup_arrays_exact": left["setup"]["arrays"] == right["setup"]["arrays"],
                          "initial_png_bytes_exact": left["setup"]["initial_image"]["sha256"] == right["setup"]["initial_image"]["sha256"],
                          "goal_png_bytes_exact": left["setup"]["goal_image"]["sha256"] == right["setup"]["goal_image"]["sha256"],
                          "noop_full_trace_bytes_exact": left["noop"]["trace"]["sha256"] == right["noop"]["trace"]["sha256"]}
                episode.update(status="completed", checks=checks, all_exact=all(checks.values()))
                require(episode["all_exact"], "Paired CPU setup/noop differs")
            except Exception as error:
                episode.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
            write(out / "summary.json", report)
            print(json.dumps({"seed": seed, "status": episode["status"], "all_exact": episode.get("all_exact")}), flush=True)
        report["cpu_prerequisites_passed"] = all(e["status"] == "completed" and e["all_exact"]
            and all(not r["setup"]["initial_success"] and r["setup"]["expert_final_success"] for r in e["repeats"])
            for e in report["episodes"])
        report["status"] = "completed_cpu_validation"
    except BaseException as error:
        report.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        report["cuda_initialized"] = torch.cuda.is_initialized()
        write(out / "summary.json", report)


def source_inventory():
    paths = {Path(__file__).resolve(), OFFICIAL, UPSTREAM / "pyproject.toml", UPSTREAM / "hubconf.py"}
    for name in ("evals/simu_env_planning", "app/vjepa_wm", "app/plan_common", "src/models"):
        paths.update((UPSTREAM / name).rglob("*.py"))
    paths.add(UPSTREAM / "evals/utils.py")
    paths.update(DINO_SOURCE.rglob("*.py"))
    for package in ("metaworld", "mujoco", "gymnasium", "gym", "clusterscope"):
        dist = importlib.metadata.distribution(package)
        for entry in dist.files or []:
            name = str(entry)
            if name.endswith((".py", ".so", ".xml", ".stl", ".obj", ".png", ".jpg")):
                p = Path(dist.locate_file(entry)).resolve()
                if p.is_file(): paths.add(p)
    paths.add(OSMESA / "libOSMesa.so.8")
    return {str(p): sha(p) for p in sorted(paths)}


def prepare_registration(config_path, cpu_report, out):
    cpu, _ = read(cpu_report)
    require(cpu["status"] == "completed_cpu_validation" and cpu["cpu_prerequisites_passed"] is True
            and cpu["runner"] == bind(__file__) and cpu["seeds"] == SEEDS, "CPU prerequisite gate not met")
    config = {"schema_version": 1, "status": "registered_before_any_model_rollout", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "runner": bind(__file__), "official_config": bind(OFFICIAL), "cpu_report": bind(cpu_report),
              "out_dir": str(out.resolve()), "seeds": SEEDS, "gate": GATE,
              "model": bind(MODEL), "dino_weights": bind(DINO_WEIGHTS), "versions": versions(),
              "source_sha256": source_inventory(),
              "cpu_artifact_sha256": {str(p.resolve()): sha(p) for p in sorted(Path(cpu_report).parent.rglob("*")) if p.is_file()},
              "protocol": {"task": "mw-reach", "total_steps": 100, "warmup_zero_controls": 1, "policy_control_budget": 99,
                  "success": "Final native info.success: TCP Euclidean distance to task target <=0.05; done_at_succ=False",
                  "expert_goal": "Native separate expert environment final visual/proprio after100 controls; never provide expert controls to planner",
                  "noop": "Raw physical [0,0,0,0], not normalized zero; two exact99-step CPU repeats per seed are frozen before Clean",
                  "cem": {"samples": 300, "iterations": 15, "elites": 10, "horizon": 6, "num_act_stepped": 3, "alpha": .1, "context": 2},
                  "model_action": "20D = five consecutive4D controls; action_skip1; no repeat; native TimeLimit truncates last chunk",
                  "expected_query_elapsed_steps": [1, 16, 31, 46, 61, 76, 91],
                  "physical_seed_rule": "rank0/local_seed1 ep0,1,2 -> (1**2+ep*1) -> seeds1,2,3",
                  "cem_rng": "One native GC_Agent CUDA Generator seeded1, uninterrupted across three Clean episodes in fixed seed order",
                  "precision": "FP32; native300 candidate batch; no autocast, microbatching, training or alternate backend",
                  "image_normalization": [[.5, .5, .5], [.5, .5, .5]], "proprio_cost_retained": True,
                  "renderer": "CPU OSMesa; no EGL", "heads_cfg": {}, "decode_each_iteration": False},
              "adaptations": ["Fresh independently rebuilt environment per episode with global initialization seed1; not a claim of bitwise reproduction of official48-episode environment reuse.",
                  "Use native eval reset/seeded_rand_vec/seed/reset_warmup/set_task expert setup. Never use MetaWorld.prepare (which ignores state) for restoration.",
                  "Reuse original GC_Agent, CEM, encoder, predictor, objective, action normalization and TimeLimit. Omit optional decoders/plots only.",
                  "Use local DINO weights and published training DATA_STATS, with strict checkpoint loading; no remote downloads or dataset requirement.",
                  "Block the unused eager training-dataset initializer while importing the unchanged native PlanEvaluator; its invocation raises. Expert goals never require that module or a remote API.",
                  "Installed environment versions are hash-pinned here and may differ from upstream dependency pins; no compatibility change is hidden."],
              "scope": "Clean normal-task readiness only. Prior negative results and failed attack gates remain unchanged.",
              "api_requests": 0, "training_updates": 0}
    write(config_path, config, exclusive=True)
    return bind(config_path)


def preflight(path):
    config, digest = read(path)
    require(config["runner"] == bind(__file__) and config["gate"] == GATE and config["seeds"] == SEEDS, "Registration changed")
    require(config["versions"] == versions() and config["model"] == bind(MODEL)
            and config["dino_weights"] == bind(DINO_WEIGHTS) and config["official_config"] == bind(OFFICIAL), "Runtime/model/config changed")
    require(config["source_sha256"] == source_inventory(), "Source/environment changed")
    for name, digest_expected in config["cpu_artifact_sha256"].items():
        require(sha(name) == digest_expected, "Frozen CPU artifact changed: " + name)
    require(bind(config["cpu_report"]["path"]) == config["cpu_report"], "CPU summary changed")
    cpu, _ = read(config["cpu_report"]["path"])
    require(cpu["cpu_prerequisites_passed"] is True, "CPU prerequisites failed")
    return config, cpu, {"status": "cpu_preflight_passed", "config": {"path": str(Path(path).resolve()), "sha256": digest},
                          "source_files": len(config["source_sha256"]), "GPU_used": False, "environment_created": False}


def run(config_path):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    config, cpu, validation = preflight(config_path)
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True  # Official eval.py setting.
    require(torch.cuda.is_available(), "CUDA required; no CPU model-rollout fallback")
    cfg = cfg_for(out, "cuda")
    summary = {"status": "loading", "config": validation["config"], "runner": bind(__file__),
               "episodes": [], "gate": None, "api_requests": 0, "training_updates": 0,
               "cuda_runtime": {"torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}
    write(out / "summary.json", summary)
    try:
        model, pre, loading = load_model(cfg, "cuda")
        summary["model_load"] = loading
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
        for seed, reference in zip(SEEDS, cpu["episodes"]):
            require(reference["seed"] == seed, "CPU reference seed order differs")
            directory = out / f"seed{seed}_clean"
            episode = {"seed": seed, "status": "preparing", "queries": []}
            summary["episodes"].append(episode)
            summary["status"] = "running"
            write(out / "summary.json", summary)
            env = None
            try:
                env, obs, goal, setup = setup_episode(cfg, seed, directory)
                ref_setup = reference["repeats"][0]["setup"]
                require(setup["arrays"] == ref_setup["arrays"] and
                        all(setup[k]["sha256"] == ref_setup[k]["sha256"] for k in ("initial_image", "goal_image")),
                        "Clean initial/goal/expert setup differs from frozen matched no-op")
                episode["setup"] = setup
                episode["matched_noop"] = reference["repeats"][0]["noop"]
                episode["initial_goal_exact"] = True
                agent.set_goal(goal)
                require(agent.planner.num_samples == 300, "Native candidate batch changed")
                rows = []
                with (directory / "trace.jsonl").open("x") as trace:
                    while env.steps_left() > 0:
                        elapsed = env.elapsed_steps()
                        steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                        tick = time.monotonic()
                        normalized = agent.act(obs, steps_left=steps_left)
                        actions = pre.denormalize_actions(normalized.cpu().reshape(-1, 4))
                        require(torch.isfinite(actions).all(), "Nonfinite native action")
                        query = {"index": len(episode["queries"]) + 1, "elapsed_before": elapsed,
                                 "planning_steps_left": steps_left, "horizon": min(6, steps_left),
                                 "normalized_macro_actions": normalized.cpu().tolist(), "physical_actions": actions.tolist(),
                                 "planned_control_count": len(actions), "executed_control_count": 0,
                                 "minimum_cost_per_iteration": agent._prev_losses.cpu().tolist(),
                                 "elapsed_seconds": time.monotonic() - tick}
                        episode["queries"].append(query)
                        for index, action in enumerate(actions):
                            visual, info, row = trace_step(env, action, query["index"], index)
                            rows.append(row); trace.write(json.dumps(row, allow_nan=False) + "\n"); trace.flush()
                            query["executed_control_count"] += 1
                            obs = TensorDict({"visual": visual, "proprio": info["proprio"]})
                            if row["done"]: break
                        write(directory / "queries.json", episode["queries"])
                        write(out / "summary.json", summary)
                        print(json.dumps({"seed": seed, "query": query["index"], "environment_steps": env.elapsed_steps(),
                                          "reach_distance": rows[-1]["reach_distance"], "success": rows[-1]["success"]}), flush=True)
                        if rows[-1]["done"]: break
                episode.update(status="completed", outcome=outcome(rows, directory))
                require([q["elapsed_before"] for q in episode["queries"]] == config["protocol"]["expected_query_elapsed_steps"], "Native replan schedule differs")
            except BaseException as error:
                episode.update(status="error", exception_type=type(error).__name__, exception=str(error))
                raise
            finally:
                if env is not None: env.close()
                write(out / "summary.json", summary)
        clean = sum(e["outcome"]["final_success"] for e in summary["episodes"])
        noop_count = sum(e["matched_noop"]["final_success"] for e in summary["episodes"])
        summary["gate"] = {"passed": clean >= 2 and clean - noop_count >= 2,
                           "clean_final_successes": clean, "noop_final_successes": noop_count,
                           "clean_minus_noop": clean - noop_count, "definition": GATE}
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("cpu-check", "prepare-registration", "preflight", "run"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cpu-report", type=Path, default=DEFAULT_CPU / "summary.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    bootstrap(cpu=not args.run)
    if args.cpu_check:
        cpu_check((args.out_dir or DEFAULT_CPU).resolve()); return
    if args.run:
        run(args.config); return
    if args.prepare_registration:
        result = prepare_registration(args.config, args.cpu_report, args.out_dir or DEFAULT_OUT)
    else:
        result = preflight(args.config)[2]
    if args.report: write(args.report, result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
