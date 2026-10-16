#!/usr/bin/env python3
"""Fixed six-process EGL rendering diagnosis; no policy/model inference or utility.

Preparation/preflight are CPU-only. --run launches exactly six fresh subprocesses,
whose GPU use is EGL rendering only. This never resumes or changes the utility v1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBERO = ROOT / "vendor/LIBERO"
CONFIG = ROOT / "configs/cosmos_rendering_reproducibility_v1.json"
OUT = ROOT / "runs/cosmos_rendering_reproducibility_v1"
SOURCE_DIR = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2"
FAIL_DIR = ROOT / "runs/cosmos_first_candidate_utility_matched30_v1/task08_init01/candidate1"
PROTOCOL = {"suite": "libero_goal", "task_id": 8, "init_ids": [0, 1], "repetitions": 3,
            "seed": 31, "env_seed": 0, "settle_steps": 10, "camera_size": 256,
            "render_gpu_device_id": 0, "MUJOCO_GL": "egl", "child_timeout_seconds": 300,
            "policy_actions": 0, "checkpoint_loads": 0, "model_forward_calls": 0,
            "training_updates": 0, "api_requests": 0}
PLAN = [{"index": 2 * r + init, "task_id": 8, "init_id": init, "repetition": r,
         "role": "previously_matching_control" if init == 0 else "previously_mismatching_case"}
        for r in range(3) for init in (0, 1)]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    content = Path(path).read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(data)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(data)
        temporary.replace(path)


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def array_record(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha256": hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()}


def exact(a, b):
    return array_record(a) == array_record(b)


def versions():
    result = {"python": sys.version, "executable": sys.executable}
    for package in ("numpy", "torch", "torchvision", "Pillow", "mujoco", "robosuite", "PyOpenGL"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def resource_inventory():
    sources = {Path(__file__).resolve(), ROOT / "scripts/pilot_cosmos_policy_native_task_gate.py",
               ROOT / "scripts/pilot_cosmos_first_candidate_utility.py", ROOT / "scripts/cosmos_policy_codecs.py",
               ROOT / "scripts/audit_cosmos_initial_observation_parity.py"}
    sources.update((LIBERO / "libero/libero").rglob("*.py"))
    for package, directory in (("robosuite", "robosuite"), ("mujoco", "mujoco")):
        location = Path(importlib.metadata.distribution(package).locate_file(directory))
        sources.update(location.rglob("*.py"))
        sources.update(location.rglob("*.so"))
    references, physical = {}, {}
    episodes = {}
    for init in (0, 1):
        stem = f"libero_goal_task08_init{init:02d}_clean"
        episode_path = SOURCE_DIR / (stem + ".json")
        episode, _ = read(episode_path)
        require(episode["status"] == "completed" and episode["condition"] == "clean"
                and episode["task_id"] == 8 and episode["init_id"] == init, "Unexpected source episode")
        episodes[str(init)] = {"episode": bind(episode_path), "goal": episode["goal"],
                               "task_name": episode["task_name"], "initial_state_sha256": episode["initial_state_sha256"]}
        for suffix in (".json", ".jsonl", "_states.npz", "_initial.png"):
            p = SOURCE_DIR / (stem + suffix)
            references[str(p)] = sha(p)
        bddl = Path(episode["bddl"])
        init_file = LIBERO / "libero/libero/init_files" / bddl.parent.name / (bddl.stem + ".pruned_init")
        require(sha(bddl) == episode["bddl_sha256"] and sha(init_file) == episode["init_file_sha256"], "Physical source changed")
        physical[str(bddl)] = sha(bddl)
        physical[str(init_file)] = sha(init_file)
        episodes[str(init)].update(bddl=str(bddl), init_file=str(init_file))
    for filename in ("initial_parity.json", "libero_goal_task08_init01_clean_initial.png", "libero_goal_task08_init01_clean.json"):
        references[str(FAIL_DIR / filename)] = sha(FAIL_DIR / filename)
    references[str(ROOT / "runs/cosmos_initial_observation_parity_cpu_v1/analysis.json")] = sha(ROOT / "runs/cosmos_initial_observation_parity_cpu_v1/analysis.json")
    return {"sources": {str(p): sha(p) for p in sorted(sources)}, "references": references,
            "physical_files": physical, "episodes": episodes}


def prepare(path, out):
    resources = resource_inventory()
    original, _ = read(SOURCE_DIR / "libero_goal_task08_init01_clean.json")
    require(sha(ROOT / "scripts/pilot_cosmos_policy_native_task_gate.py") == original["provenance"]["runner_sha256"], "Original native runner changed")
    config = {"schema_version": 1, "scope": "bounded_environment_rendering_reproducibility_only",
              "registered_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prospective_not_executed",
              "runner": bind(__file__), "out_dir": str(Path(out).resolve()), "protocol": PROTOCOL, "planned_trials": PLAN,
              "resources": resources, "versions": versions(),
              "native_order": ["python/numpy/torch/CUDA seed31; cudnn deterministic true/benchmark false",
                               "construct OffScreenRenderEnv original BDDL, camera256, render_gpu_device_id0",
                               "env.seed(0)", "env.reset()", "env.set_init_state(original_init)",
                               "get initial sim state and copy existing observations", "10 original dummy actions [0,0,0,0,0,0,-1]",
                               "get settled sim state and copy existing observations", "check_success; save images; no policy query"],
              "budget_policy": "Exactly six subprocess invocations, fixed alternating init0/init1 order, each with a 300-second timeout. Save all matches, mismatches, exceptions and timeouts. No replacement, pixel-based retry, best-of selection or extension. A provenance failure before execution blocks the run.",
              "native_internal_reset": "Original env.reset may internally retry RandomizationError. Preserve native method, record the underlying reset-call count/errors after construction; child timeout bounds execution. Constructor internals are not claimed fully counted.",
              "gpu_scope": "GPU is used for native EGL rendering only; torch CUDA seed setting is retained but no CUDA tensor/model computation or checkpoint load occurs.",
              "comparisons": ["initial/settled sim arrays and settled first EEF vs old source", "raw combined PNG bytes and decoded pixels per view vs source",
                              "Task8Init1 vs already saved failing image", "all three repeats within each init; no sample selection"],
              "state_scope": "Archive explicitly listed simulator/controller/camera/material/render-context numerical fields, cached raw observations and Python/NumPy/torch-CPU RNG. This is an expanded diagnostic signature, not proof of complete engine/driver/controller state. Do not force another render or forward merely to collect diagnostics.",
              "interpretation": "Cold fresh subprocesses do not reproduce the previous process's 50-environment/model-allocation history. Matching cannot establish absence of order/history effects. Mismatching does not establish a physical task difference or rendering root cause. No task utility, attack, defense, threshold relaxation, or restoration of terminal v1 is authorized by this diagnostic.",
              "checkpoint_loaded": False, "model_forward_calls": 0, "training_updates": 0, "api_requests": 0}
    write(path, config, exclusive=True)
    return bind(path)


def preflight(path):
    config, digest = read(path)
    require(config["scope"] == "bounded_environment_rendering_reproducibility_only" and config["schema_version"] == 1, "Wrong scope")
    require(config["protocol"] == PROTOCOL and config["planned_trials"] == PLAN, "Fixed budget/protocol changed")
    require(config["runner"] == bind(__file__), "Runner changed after registration")
    require(config["versions"] == versions(), "Registered Python/package versions changed")
    require(config["resources"] == resource_inventory(), "Resource catalog or hash changed")
    return config, {"status": "cpu_preflight_passed", "config": {"path": str(Path(path).resolve()), "sha256": digest},
                    "source_files": len(config["resources"]["sources"]), "planned_children": 6,
                    "environment_created": False, "GPU_used_in_preflight": False}


def numeric_fields(obj, prefix, arrays, scalars, depth=0):
    """Copy plain stored values only: no method calls or descriptor discovery."""
    try:
        fields = vars(obj)
    except TypeError:
        return
    for key, value in sorted(fields.items()):
        name = prefix + "." + key
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            arrays[name] = value.copy()
        elif value is None or type(value) in (str, bool, int, float) or isinstance(value, np.number):
            value = value.item() if isinstance(value, np.generic) else value
            if not isinstance(value, float) or np.isfinite(value):
                scalars[name] = value
        elif depth and key in ("interpolator_pos", "interpolator_ori", "gripper", "controller"):
            numeric_fields(value, name, arrays, scalars, depth - 1)


def snapshot(env, obs):
    arrays = {"sim": np.asarray(env.get_sim_state()).copy()}
    scalars, unavailable = {}, []
    def capture(obj, names, prefix):
        for name in names:
            try:
                a = np.asarray(getattr(obj, name))
                if a.dtype.kind not in "biufc":
                    raise TypeError("non-numeric field")
                arrays[prefix + "." + name] = a.copy()
            except (AttributeError, TypeError, ValueError) as error:
                unavailable.append({"field": prefix + "." + name, "reason": type(error).__name__})
    capture(env.sim.data, ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "qfrc_applied", "xfrc_applied",
                          "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "geom_xpos", "geom_xmat", "ncon", "time"), "data")
    capture(env.sim.model, ("body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy", "cam_ipd", "light_pos", "light_dir", "light_ambient", "light_diffuse", "light_specular",
                           "geom_rgba", "geom_matid", "mat_rgba", "mat_emission", "mat_specular", "mat_shininess", "mat_reflectance", "tex_rgb"), "model")
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            arrays["obs." + key] = value.copy()
    for i, robot in enumerate(env.robots):
        numeric_fields(robot.controller, f"robot{i}.controller", arrays, scalars, depth=1)
        numeric_fields(robot.gripper, f"robot{i}.gripper", arrays, scalars)
    numeric_fields(env.env, "environment", arrays, scalars)
    for name, observable in getattr(env.env, "_observables", {}).items():
        numeric_fields(observable, "observable." + name, arrays, scalars)
    for name, value in getattr(env.env, "_obs_cache", {}).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            arrays["observation_cache." + name] = value.copy()
        elif type(value) in (bool, int, float) and np.isfinite(value):
            scalars["observation_cache." + name] = value
    context = getattr(env.sim, "_render_context_offscreen", None)
    if context is not None:
        numeric_fields(context, "render", arrays, scalars)
        for key, fields in {
            "cam": ("type", "fixedcamid", "trackbodyid", "lookat", "distance", "azimuth", "elevation"),
            "vopt": ("flags", "geomgroup", "sitegroup", "jointgroup", "tendongroup", "actuatorgroup", "label", "frame"),
            "scn": ("flags", "ngeom", "nlight", "maxgeom", "enabletransform", "translate", "rotate", "scale", "stereo"),
            "con": ("offWidth", "offHeight", "offSamples", "currentBuffer", "glInitialized", "shadowSize"),
        }.items():
            obj = getattr(context, key, None)
            if obj is not None:
                capture(obj, fields, "render." + key)
        scn = getattr(context, "scn", None)
        if scn is not None:
            for key in ("pos", "mat", "size", "rgba", "type", "dataid", "objtype", "objid", "emission", "specular", "shininess", "reflectance"):
                try:
                    value = np.asarray([getattr(scn.geoms[i], key) for i in range(scn.ngeom)])
                    if value.dtype.kind not in "biufc":
                        raise TypeError("non-numeric scene geometry field")
                    arrays["render.geoms." + key] = value.copy()
                except (AttributeError, TypeError, ValueError) as error:
                    unavailable.append({"field": "render.geoms." + key, "reason": type(error).__name__})
    return arrays, {"scalars": scalars, "unavailable_fields": unavailable,
                    "scope": "Listed numerical fields only; no full driver/controller-state equivalence claim"}


def archive_snapshot(directory, label, arrays, metadata):
    path = directory / (label + ".npz")
    np.savez_compressed(path, **arrays)
    result = {**metadata, "arrays": {key: array_record(value) for key, value in arrays.items()}, "archive": bind(path)}
    write(directory / (label + ".json"), result, exclusive=True)
    return result


def save_images(directory, label, arrays):
    from PIL import Image
    primary, wrist = [arrays["obs." + key] for key in ("agentview_image", "robot0_eye_in_hand_image")]
    outputs = {}
    for view, value in (("primary_raw", primary), ("wrist_raw", wrist),
                        ("combined_native", np.concatenate([np.ascontiguousarray(primary[::-1]), np.ascontiguousarray(wrist[::-1])], axis=1))):
        path = directory / f"{label}_{view}.png"
        Image.fromarray(value).save(path)
        outputs[view] = bind(path)
    return outputs


def compare_image_files(source, actual):
    from PIL import Image
    from audit_cosmos_initial_observation_parity import compare
    a, b = [np.asarray(Image.open(path)) for path in (source, actual)]
    return {"source": bind(source), "actual": bind(actual), "png_bytes_exact": sha(source) == sha(actual),
            "views": {view: compare(a[:, offset:offset + 256], b[:, offset:offset + 256])
                      for view, offset in (("primary", 0), ("wrist", 256))}}


def child(config_path, index):
    config, verification = preflight(config_path)
    require(type(index) is int and 0 <= index < 6, "Unregistered child")
    item = PLAN[index]
    directory = Path(config["out_dir"]) / f"trial{index:02d}_init{item['init_id']}_repeat{item['repetition']}"
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "starting", **item, "config": verification["config"], "runner": config["runner"],
              "pid": os.getpid(), "versions": versions(), "gpu_use": "EGL rendering only; no model computation",
              "checkpoint_loaded": False, "model_forward_calls": 0, "training_updates": 0, "api_requests": 0,
              "explicit_settling_steps": 0, "policy_steps": 0}
    write(directory / "result.json", result)
    env = None
    started = time.monotonic()
    try:
        require(os.environ.get("MUJOCO_GL", "egl") == "egl", "EGL backend required")
        require(os.environ.get("PYOPENGL_PLATFORM", "egl") == "egl", "Unexpected PyOpenGL platform")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
        sys.path.insert(0, str(LIBERO))
        import torch
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import benchmark
        task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
        source = config["resources"]["episodes"][str(item["init_id"])]
        require(task.language == source["goal"] and task.name == source["task_name"], "Original task mapping changed")
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        torch.cuda.manual_seed_all(31)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        init_states = torch.load(source["init_file"], map_location="cpu", weights_only=False)
        initial = np.asarray(init_states[item["init_id"]])
        require(hashlib.sha256(initial.tobytes()).hexdigest() == source["initial_state_sha256"], "Initial state file/index mismatch")
        env = OffScreenRenderEnv(bddl_file_name=source["bddl"], camera_heights=256, camera_widths=256, render_gpu_device_id=0)
        env.task_id, env.task_name, env.task_suite_name = 8, task.name, "libero_goal"
        reset_events = []
        original_reset = env.env.reset
        def counted_reset(*args, **kwargs):
            event = {"call": len(reset_events) + 1}
            reset_events.append(event)
            try:
                value = original_reset(*args, **kwargs)
                event["status"] = "returned"
                return value
            except BaseException as error:
                event.update(status="error", error_type=type(error).__name__, error=str(error))
                raise
        env.env.reset = counted_reset
        env.seed(0)
        env.reset()
        obs = env.set_init_state(init_states[item["init_id"]])
        initial_arrays, initial_metadata = snapshot(env, obs)
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
            result["explicit_settling_steps"] += 1
        settled_arrays, settled_metadata = snapshot(env, obs)
        result["success_at_start_diagnostic_only"] = bool(env.check_success())
        result["reset_events_after_construction"] = reset_events
        result["snapshots"] = {"initial": archive_snapshot(directory, "initial", initial_arrays, initial_metadata),
                               "settled": archive_snapshot(directory, "settled", settled_arrays, settled_metadata)}
        result["images"] = {"initial": save_images(directory, "initial", initial_arrays),
                            "settled": save_images(directory, "settled", settled_arrays)}
        rng_path = directory / "cpu_rng_states.pkl"
        with rng_path.open("xb") as handle:
            pickle.dump({"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch_cpu": torch.get_rng_state().numpy()}, handle, protocol=5)
        result["cpu_rng_states"] = bind(rng_path)
        source_episode, _ = read(source["episode"]["path"])
        state_path = Path(source_episode["trace"]).with_name(Path(source_episode["trace"]).stem + "_states.npz")
        with np.load(state_path, allow_pickle=False) as prior:
            result["source_state_comparison"] = {"initial_sim_exact": exact(initial_arrays["sim"], prior["initial"]),
                                                  "settled_sim_exact": exact(settled_arrays["sim"], prior["settled"]),
                                                  "source": bind(state_path)}
        with Path(source_episode["trace"]).open() as handle:
            first = json.loads(next(handle))
        result["source_state_comparison"]["settled_eef_exact"] = exact(settled_arrays["obs.robot0_eef_pos"], np.asarray(first["eef_before"], dtype=np.float64))
        actual_png = result["images"]["settled"]["combined_native"]["path"]
        result["source_image_comparison"] = compare_image_files(source_episode["initial_image"], actual_png)
        if item["init_id"] == 1:
            result["failure_image_comparison"] = compare_image_files(FAIL_DIR / "libero_goal_task08_init01_clean_initial.png", actual_png)
        result["runtime_environment"] = {key: os.environ.get(key) for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "LIBERO_CONFIG_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")}
        result["python_hash_randomization"] = {"enabled": sys.flags.hash_randomization,
                                               "fixed_string_hash_diagnostic": hash("cosmos_rendering_reproducibility")}
        try:
            from OpenGL import GL
            result["opengl"] = {name: (value.decode(errors="replace") if value else None)
                                for name, value in (("vendor", GL.glGetString(GL.GL_VENDOR)),
                                                    ("renderer", GL.glGetString(GL.GL_RENDERER)), ("version", GL.glGetString(GL.GL_VERSION)))}
        except Exception as error:
            result["opengl"] = {"read_error": str(error), "error_type": type(error).__name__}
        result["torch_cuda_initialized"] = torch.cuda.is_initialized()
        result["status"] = "completed_diagnostic"
    except BaseException as error:
        result.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(directory / "result.json", result)
        if env is not None:
            env.close()


def run(config_path):
    config, verification = preflight(config_path)
    output = Path(config["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Fresh output required; resuming/replacing a trial is forbidden")
    write(output / "preflight.json", verification, exclusive=True)
    summary = {"status": "running", "config": verification["config"], "protocol": PROTOCOL,
               "planned_trials": PLAN, "trials": [], "gpu_use": "EGL rendering only in six fresh child processes",
               "utility_conclusions": None, "model_forward_calls": 0, "training_updates": 0, "api_requests": 0}
    write(output / "summary.json", summary)
    for item in PLAN:
        directory = output / f"trial{item['index']:02d}_init{item['init_id']}_repeat{item['repetition']}"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--child-index", str(item["index"]), "--config", str(Path(config_path).resolve())]
        log_path = output / f"trial{item['index']:02d}.log"
        row = {**item, "command": command, "status": "invoked"}
        started = time.monotonic()
        with log_path.open("x") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=300, check=False, cwd=ROOT)
                row.update(status="returned", exit_code=completed.returncode)
            except subprocess.TimeoutExpired:
                row.update(status="timeout", exit_code=None)
        row.update(elapsed_seconds=time.monotonic() - started, log=bind(log_path))
        if (directory / "result.json").exists():
            result, digest = read(directory / "result.json")
            row["result"] = {"path": str(directory / "result.json"), "sha256": digest, "status": result["status"]}
        summary["trials"].append(row)
        write(output / "summary.json", summary)
        print(json.dumps({"completed_invocations": len(summary["trials"]), "index": item["index"], "subprocess_status": row["status"], "exit_code": row["exit_code"]}), flush=True)
    require(len(summary["trials"]) == 6, "Fixed invocation budget not completed")
    summary["status"] = "completed_diagnostic" if all(t.get("exit_code") == 0 and t.get("result", {}).get("status") == "completed_diagnostic" for t in summary["trials"]) else "completed_with_errors"
    summary["within_init_comparisons"] = []
    for init in (0, 1):
        available = [t for t in summary["trials"] if t["init_id"] == init and t.get("result", {}).get("status") == "completed_diagnostic"]
        for left_index in range(len(available)):
            for right_index in range(left_index + 1, len(available)):
                left, _ = read(available[left_index]["result"]["path"])
                right, _ = read(available[right_index]["result"]["path"])
                summary["within_init_comparisons"].append({"left_trial": left["index"], "right_trial": right["index"],
                    "settled_image": compare_image_files(left["images"]["settled"]["combined_native"]["path"], right["images"]["settled"]["combined_native"]["path"]),
                    "numeric_snapshot_field_signatures_exact": {stage: left["snapshots"][stage]["arrays"] == right["snapshots"][stage]["arrays"] for stage in ("initial", "settled")}})
    summary["error_policy"] = "All six invocations retained; incomplete/error trials are not replaced and do not disappear from the denominator. Pair comparisons require both image artifacts, with missing trials visible above. No task utility/gate conclusion."
    write(output / "summary.json", summary)


def self_test():
    checks = {"exactly_six": len(PLAN) == 6, "three_per_init": all(sum(t["init_id"] == i for t in PLAN) == 3 for i in (0, 1)),
              "fixed_order": [t["init_id"] for t in PLAN] == [0, 1, 0, 1, 0, 1],
              "unique_trials": len({(t["init_id"], t["repetition"]) for t in PLAN}) == 6,
              "exact_dtype": not exact(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64)),
              "exact_signed_zero": not exact(np.array([0.]), np.array([-0.])),
              "no_model_or_environment_import": "torch" not in sys.modules and "libero.libero.envs" not in sys.modules,
              "zero_policy_budget": PROTOCOL["policy_actions"] == PROTOCOL["model_forward_calls"] == 0}
    require(all(checks.values()), "CPU schedule/hash test failed")
    return {"status": "passed", "checks": checks, "GPU_used": False, "environment_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-registration", type=Path)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.prepare_registration:
        result = prepare(args.prepare_registration, args.out_dir)
    elif args.preflight:
        result = preflight(args.config)[1]
    elif args.self_test:
        result = self_test()
    elif args.child:
        child(args.config, args.child_index)
        return
    else:
        run(args.config)
        return
    if args.report:
        write(args.report, result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
