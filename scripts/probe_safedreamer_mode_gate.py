#!/usr/bin/env python3
"""Frozen SafeDreamer OSRP first-root resource and mode feasibility preflight.

This entry stops after one native six-round query. It never executes an action,
changes sampling, runs a Clean episode, or interprets a first query as competence.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/safedreamer_mode_mediator_gate_20261016.json"
SOURCE = ROOT / "vendor/SafeDreamer"
CHECKPOINT = ROOT / "models/SafeDreamer/osrp_pointgoal1_seed0.ckpt"


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def gpu_rows():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=5)
    return [dict(zip(("index", "uuid", "total_mib", "free_mib"),
                     [int(p.strip()) if i != 1 else p.strip()
                      for i, p in enumerate(line.split(","))]))
            for line in raw.strip().splitlines()]


def own_gpu_mib():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=5)
    return sum(int(parts[1].strip()) for line in raw.strip().splitlines()
               if len(parts := line.split(",")) == 2
               and parts[0].strip() == str(os.getpid())
               and parts[1].strip().isdigit())


def install_audit_methods(agt, nets, behaviors, nj):
    """Append observation leaves to unchanged native arithmetic; no RNG draws."""
    def compile_method(cls, name, transform):
        original = inspect.unwrap(getattr(cls, name))
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        fn = tree.body[0]
        fn.decorator_list = []
        transform(fn)
        ast.fix_missing_locations(tree)
        ns = dict(original.__globals__)
        exec(compile(tree, f"<audit:{cls.__name__}.{name}>", "exec"), ns)
        setattr(cls, name, nj._scope_method(ns[name]))

    def prior(fn):
        found = 0
        body = []
        for node in fn.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "stoch"):
                assert ast.unparse(node.value) == "dist.sample(seed=nj.rng())"
                body.extend(ast.parse("_audit_key = nj.rng()").body)
                node.value.keywords[0].value = ast.Name(id="_audit_key", ctx=ast.Load())
                found += 1
            if isinstance(node, ast.Return):
                assert ast.unparse(node.value) == "cast(prior)"
                body.extend(ast.parse(
                    "_audit_result = cast(prior)\n"
                    "_audit_result['_audit_prior_key'] = _audit_key\n"
                    "return _audit_result").body)
            else:
                body.append(node)
        assert found == 1
        fn.body = body

    def imagine(fn):
        class Start(ast.NodeTransformer):
            count = 0
            def visit_Assign(self, node):
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "start" and isinstance(node.value, ast.DictComp)):
                    self.count += 1
                    return [node, *ast.parse(
                        "start['_audit_prior_key'] = jnp.zeros((2,), dtype=jnp.uint32)").body]
                return node
        obj = Start()
        obj.visit(fn)
        assert obj.count == 2

    def planner(fn):
        body = []
        count = 0
        for node in fn.body:
            if isinstance(node, ast.For) and ast.unparse(node.target) == "i":
                body.extend(ast.parse("_audit_rounds = []").body)
                node.body[0:0] = ast.parse(
                    "_audit_mean_before = mean\n_audit_std_before = std\n"
                    "_audit_rng_before = nj.context().rng\n"
                    "_audit_reserve_before = tuple(nj.context().reserve)").body
                node.body.extend(ast.parse("""
_audit_round = {
    'mean_before': _audit_mean_before, 'std_before': _audit_std_before,
    'rng_before': _audit_rng_before, 'reserve_before': _audit_reserve_before,
    'actions': traj_pi_gaus['action'], 'costs': traj_cost,
    'returns': traj_rew, 'safe_count': num_safe_traj,
    'elite_indices_in_native_filtered_array': elite_idxs,
    'mean_after': mean, 'std_after': std,
}
if i == self.iterations - 1:
    _audit_round['trajectory'] = traj_pi_gaus
_audit_rounds.append(_audit_round)
""").body)
                count += 1
            if isinstance(node, ast.Return):
                assert isinstance(node.value, ast.Tuple)
                assert isinstance(node.value.elts[0], ast.Dict)
                node.value.elts[0].keys.append(ast.Constant("_audit_rounds"))
                node.value.elts[0].values.append(ast.Call(
                    func=ast.Name(id="tuple", ctx=ast.Load()),
                    args=[ast.Name(id="_audit_rounds", ctx=ast.Load())], keywords=[]))
            body.append(node)
        assert count == 1
        fn.body = body

    compile_method(nets.RSSM, "img_step", prior)
    compile_method(agt.WorldModel, "imagine", imagine)
    compile_method(behaviors.CCEPlanner, "policy", planner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "gpu" and os.environ.get("SAFEDREAMER_CUDA_PATHS_READY") != "1":
        # glibc reads LD_LIBRARY_PATH at process startup, before Python imports.
        nvidia = Path(sys.prefix) / "lib/python3.9/site-packages/nvidia"
        directories = sorted(str(p) for p in nvidia.glob("*/lib") if p.is_dir())
        runtime_env = os.environ.copy()
        runtime_env["LD_LIBRARY_PATH"] = ":".join(directories + [runtime_env.get("LD_LIBRARY_PATH", "")])
        cuda_data = nvidia / "cuda_nvcc"
        runtime_env["PATH"] = str(cuda_data / "bin") + ":" + runtime_env.get("PATH", "")
        runtime_env["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + str(cuda_data)
        runtime_env["SAFEDREAMER_CUDA_PATHS_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], runtime_env)
    protocol = json.loads(CONFIG.read_text())
    out = args.output or ROOT / protocol["output"] / "native_query_attempt1"
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen_protocol.json").write_bytes(CONFIG.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    summary = {
        "status": "starting", "protocol": str(CONFIG.relative_to(ROOT)),
        "backend": args.backend, "source_revision": protocol["source_revision"],
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "stage": "resource_preflight",
        "activity": {"checkpoint_load_attempts": 0, "checkpoint_loads": 0,
                     "environment_constructions_attempted": 0,
                     "environment_constructions_completed": 0,
                     "environment_resets_attempted": 0, "environment_resets": 0,
                     "environment_steps": 0, "native_policy_queries": 0,
                     "native_policy_queries_completed": 0,
                     "policy_state_initializations": 0,
                     "policy_state_initializations_completed": 0,
                     "native_cem_rounds": 0, "imagined_candidate_transitions": 0,
                     "counterfactual_queries": 0, "training_updates": 0,
                     "model_api_calls": 0},
        "clean_competence_measured": False, "attack_measured": False,
        "mode_gate_passed": None, "peak_process_gpu_mib": 0,
        "adapters": [
            "Native inner Agent + nj.pure(create=False); no dummy training initialization",
            "Consume and record the constructor's one NumPy RNG draw before the native first policy draw",
            "AST adds only prior RNG and planner audit output leaves; native expressions retained",
            "Seed injected at the actual first underlying reset, no preliminary reset",
            "No eval reporting/video inference and no action execution"],
    }
    stop = threading.Event()
    env = None
    def update():
        summary["elapsed_seconds"] = time.monotonic() - start
        save_json(out / "summary.json", summary)
    def timed_out(signum, frame):
        raise TimeoutError("Frozen first-query 180 second resource limit")
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, protocol["resource_limits"]["first_query_wall_seconds_max"])
    update()
    try:
        if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size != protocol["checkpoint"]["bytes"]:
            raise FileNotFoundError("Exact published checkpoint size unavailable")
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.backend == "gpu" else ""
        os.environ["JAX_PLATFORM_NAME"] = args.backend
        os.environ.setdefault("MUJOCO_GL", "egl" if args.backend == "gpu" else "osmesa")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        if args.backend == "gpu":
            gpu = next(row for row in gpu_rows() if row["index"] == 0)
            summary["gpu_before"] = gpu
            if gpu["free_mib"] < protocol["resource_limits"]["gpu_free_required_mib"]:
                summary["status"] = "resource_deferred"
                return 2
            fraction = min(5 * 1024 / gpu["total_mib"], 0.20)
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
            summary["allocator_fraction"] = fraction
            summary["allocator_budget_mib"] = fraction * gpu["total_mib"]
            def monitor():
                while not stop.wait(0.5):
                    try:
                        used = own_gpu_mib()
                        summary["peak_process_gpu_mib"] = max(summary["peak_process_gpu_mib"], used)
                        if used > 6144:
                            summary["resource_limit_reason"] = "process_gpu_memory_above_6GiB"
                            os.kill(os.getpid(), signal.SIGALRM)
                            return
                    except (subprocess.SubprocessError, ValueError):
                        pass
            threading.Thread(target=monitor, daemon=True).start()
        def deny_network(*a, **kw):
            raise RuntimeError("Network disabled inside scientific probe")
        socket.create_connection = deny_network
        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network
        sys.path.insert(0, str(SOURCE))
        sys.path.insert(0, str(SOURCE / "SafeDreamer"))
        import numpy as np
        import jax
        import jax.numpy as jnp
        import embodied
        from SafeDreamer import agent as agt, behaviors, nets, ninjax as nj, jaxutils
        from SafeDreamer.train import make_env
        from embodied.envs.from_gymnasium import FromGymnasium
        summary["versions"] = {name: importlib.metadata.version(name) for name in (
            "jax", "jaxlib", "numpy", "optax", "tensorflow-probability",
            "safety-gymnasium", "gymnasium", "mujoco")}
        env_origin = importlib.metadata.distribution("safety-gymnasium").read_text("direct_url.json")
        summary["safety_gymnasium_origin"] = json.loads(env_origin) if env_origin else None
        summary["safety_gymnasium_training_revision_known"] = False
        if not env_origin or "ae966e511b9927f06b39c727ca5c650136a4e696" not in env_origin:
            raise RuntimeError("Official pinned Safety-Gymnasium source with vision_back is required")
        assert summary["versions"]["jax"] == "0.3.25"
        jax.config.update("jax_platform_name", args.backend)
        jax.config.update("jax_transfer_guard", "disallow")
        jaxutils.COMPUTE_DTYPE = jnp.float16
        cfg = embodied.Config(agt.Agent.configs["defaults"]).update(agt.Agent.configs["osrp"])
        cfg = cfg.update({"jax.platform": args.backend, "seed": 0,
                          "envs.parallel": "none", "envs.restart": False})
        assert (cfg.planner.horizon, cfg.planner.num_samples, cfg.planner.num_elites,
                cfg.planner.iterations, cfg.planner.mixture_coef, cfg.cost_limit,
                cfg.rssm.stoch, cfg.rssm.classes, cfg.rssm.unimix) == (15, 500, 50, 6, 0, 2, 48, 48, .01)
        cfg.save(embodied.Path(str(out / "native_config.yaml")))
        summary["stage"] = "checkpoint_cpu_tree_validation"
        summary["activity"]["checkpoint_load_attempts"] += 1
        update()
        with CHECKPOINT.open("rb") as fh:
            checkpoint = pickle.load(fh)
        summary["activity"]["checkpoint_loads"] += 1
        variables = checkpoint["agent"]
        assert isinstance(variables, dict) and variables
        leaf_manifest = {}
        for key, val in variables.items():
            leaves, structure = jax.tree_util.tree_flatten(val)
            rows = []
            for leaf in leaves:
                arr = np.asarray(leaf)
                if arr.dtype.kind not in "biufc" or not np.isfinite(arr).all():
                    raise ValueError(f"Invalid checkpoint leaf: {key}")
                rows.append({"shape": list(arr.shape), "dtype": str(arr.dtype)})
            leaf_manifest[key] = {"tree_structure": str(structure), "leaves": rows}
        save_json(out / "checkpoint_tree.json", leaf_manifest)
        del checkpoint
        summary["stage"] = "actual_first_seeded_reset"
        update()
        original_reset = FromGymnasium.step
        def seeded_step(self, action):
            if action["reset"] or self._done:
                if summary["activity"]["environment_resets"]:
                    raise RuntimeError("A second physical reset is forbidden")
                raw_reset = self._env.reset
                def first_reset(*a, **kw):
                    if "seed" in kw:
                        raise RuntimeError("Unexpected seed supplied by native wrapper")
                    kw["seed"] = protocol["fixed_instance"]["physical_reset_seed"]
                    summary["activity"]["environment_resets_attempted"] += 1
                    update()
                    obs_info = raw_reset(*a, **kw)
                    summary["activity"]["environment_resets"] += 1
                    return obs_info
                self._env.reset = first_reset
                try:
                    return original_reset(self, action)
                finally:
                    self._env.reset = raw_reset
            raise RuntimeError("Physical action execution is outside first-query preflight")
        FromGymnasium.step = seeded_step
        summary["activity"]["environment_constructions_attempted"] += 1
        update()
        env = make_env(cfg, mode="eval")
        summary["activity"]["environment_constructions_completed"] += 1
        summary["observation_space"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                        for k, v in env.obs_space.items()}
        summary["action_space"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                       "low": np.asarray(v.low).tolist(), "high": np.asarray(v.high).tolist()}
                                   for k, v in env.act_space.items()}
        obs = env.step({name: np.zeros(space.shape, dtype=space.dtype)
                        if name != "reset" else np.asarray(True)
                        for name, space in env.act_space.items()})
        assert bool(obs["is_first"]) and obs["image"].shape == (64, 64, 3)
        assert obs["image2"].shape == (64, 64, 3)
        np.savez_compressed(out / "initial_observation.npz", **obs)
        summary["observation_shapes"] = {k: list(np.shape(v)) for k, v in obs.items()}
        used = {}
        original_get = nj.Module.get
        def strict_get(self, name, *a, **kw):
            key = self.path + "/" + name
            if key in nj.context():
                value = nj.context()[key]
                expected = a[1] if len(a) > 1 else None
                if isinstance(expected, int):
                    expected = (expected,)
                if isinstance(expected, (tuple, list)) and all(isinstance(x, int) for x in expected):
                    if tuple(value.shape) != tuple(expected):
                        raise ValueError(f"Inference parameter shape mismatch: {key}")
                used[key] = leaf_manifest[key]
            return original_get(self, name, *a, **kw)
        nj.Module.get = strict_get
        install_audit_methods(agt, nets, behaviors, nj)
        native = agt.Agent.inner(env.obs_space, env.act_space, embodied.Counter(), cfg, name="agent")
        rng = np.random.default_rng(0)
        skipped_init_rng = int(rng.integers(2 ** 63 - 1))
        policy_rng = int(rng.integers(2 ** 63 - 1))
        summary["native_rng"] = {"constructor_draw_consumed": skipped_init_rng,
                                 "first_policy_draw": policy_rng,
                                 "post_draw_numpy_state": rng.bit_generator.state}
        device = jax.devices(args.backend)[0]
        variables = jax.device_put(variables, device)
        key = jax.device_put(np.asarray(policy_rng, dtype=np.int64), device)
        obs_batch = jax.device_put({k: np.asarray(v)[None] for k, v in obs.items()}, device)
        init = jax.jit(lambda v, k: nj.pure(lambda: native.policy_initial(1))(
            v, k, create=False), device=device)
        summary["activity"]["policy_state_initializations"] = 1
        policy_state, same_variables = init(variables, key)
        summary["activity"]["policy_state_initializations_completed"] = 1
        assert set(same_variables) == set(variables)
        def run(v, k, observation, state):
            return nj.pure(native.policy)(v, k, observation, state, mode="eval", create=False)
        run = jax.jit(run, device=device)
        summary["stage"] = "first_native_policy_query"
        summary["activity"]["native_policy_queries"] = 1
        summary["activity"]["native_cem_rounds"] = None
        summary["activity"]["imagined_candidate_transitions"] = None
        summary["partial_count_rule"] = "An interrupted JIT query has unknown completed inner rounds, never zero by assumption"
        update()
        query_start = time.monotonic()
        (outputs, policy_state), updated = run(variables, key, obs_batch, policy_state)
        outputs, policy_state = jax.device_get((outputs, policy_state))
        summary["activity"]["native_policy_queries_completed"] = 1
        summary["query_wall_seconds"] = time.monotonic() - query_start
        summary["activity"]["native_cem_rounds"] = 6
        summary["activity"]["imagined_candidate_transitions"] = 6 * 500 * 15
        rounds = outputs.pop("_audit_rounds")
        assert len(rounds) == 6
        with (out / "native_first_query.pkl").open("wb") as fh:
            pickle.dump({"outputs": outputs, "policy_state": policy_state,
                         "rounds": rounds, "rng": summary["native_rng"]}, fh)
        save_json(out / "inference_consumed_keys.json", used)
        last = rounds[-1]
        actions = np.asarray(last["actions"])
        assert actions.shape == (16, 500, 2)
        assert last["trajectory"]["_audit_prior_key"].shape == (16, 2)
        assert last["trajectory"]["_audit_prior_key"].dtype == np.uint32
        count = int(last["safe_count"])
        elite = np.asarray(last["elite_indices_in_native_filtered_array"])
        padded = bool(count >= 50 and np.any(elite >= count))
        summary["native_final_round"] = {
            "safe_count": count, "mode": "minimum_cost" if count < 50 else "safe_reward",
            "selected_padding": padded, "returned_action": outputs["action"].tolist(),
            "minimum_cost": float(np.min(last["costs"])),
            "maximum_cost": float(np.max(last["costs"])),
            "inference_consumed_keys": len(used),
        }
        summary["status"] = "stopped_native_padding_contamination" if padded else "completed_native_first_query_preflight"
        summary["next_stage"] = "stop" if padded else "independent_replay_calibration_and_fixed_shared_sampler_gate_not_executed"
        if padded:
            summary["mode_gate_passed"] = False
        summary["stage"] = "terminal"
        return 0
    except BaseException as exc:
        summary["status"] = "resource_deferred" if isinstance(exc, TimeoutError) else "technical_error"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc),
                            "traceback": traceback.format_exc()}
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                summary["close_error"] = repr(exc)
        update()


if __name__ == "__main__":
    raise SystemExit(main())
