#!/usr/bin/env python3
"""Two frozen OSRP final-round replays after a qualified native query.

Calibrate the full independent trajectory exactly before the fixed common-key
intervention. This entry never creates an environment or executes an action.
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
import sys
import textwrap
import threading
import time
import traceback

from probe_safedreamer_mode_gate import (
    CHECKPOINT, CONFIG, ROOT, SOURCE, gpu_rows, install_audit_methods,
    own_gpu_mib, save_json,
)


class CalibrationFailure(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise CalibrationFailure(message)


def compile_replay(planner_class, nj):
    """Use the unchanged original native final-loop arithmetic, not a new scorer."""
    original = inspect.unwrap(planner_class.policy)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    loops = [node for node in tree.body[0].body
             if isinstance(node, ast.For) and ast.unparse(node.target) == "i"]
    require(len(loops) == 1, "Native CCE loop changed")
    prefix = ast.parse("""
def replay_final(self, latent, mean, std, rng_before, reserve_before):
    num_pi_trajs = 0
    _mean_before = mean
    nj.context().rng = rng_before
    nj.context().reserve = list(reserve_before)
""")
    fn = prefix.body[0]
    fn.body += copy.deepcopy(loops[0].body)
    fn.body += ast.parse("""
_cost_only_ids = jax.lax.top_k(-traj_cost, self.num_elites)[1]
_cost_only_actions = traj_pi_gaus['action'][:, _cost_only_ids, :]
_cost_only_mean = self.momentum * _mean_before + (1 - self.momentum) * _cost_only_actions.mean(axis=1)
_cost_only_std = _cost_only_actions.std(axis=1)
return {'trajectory': traj_pi_gaus, 'actions': traj_pi_gaus['action'],
        'costs': traj_cost, 'returns': traj_rew, 'safe_count': num_safe_traj,
        'elite_indices_in_native_filtered_array': elite_idxs,
        'mean_after': mean, 'std_after': std, 'action': mean[0],
        'always_minimum_cost': {'elite_original_candidate_ids': _cost_only_ids,
                               'mean_after': _cost_only_mean, 'std_after': _cost_only_std,
                               'action': _cost_only_mean[0]},
        'rng_after': nj.context().rng, 'reserve_after': tuple(nj.context().reserve)}
""").body
    ast.fix_missing_locations(prefix)
    namespace = dict(original.__globals__)
    exec(compile(prefix, "<frozen-native-final-CCE-loop>", "exec"), namespace)
    return nj._scope_method(namespace["replay_final"]), ast.unparse(prefix)


def compile_common_prior(rssm_class, nj):
    """Change only prior categorical sampling, preserving original NN expressions."""
    original = inspect.unwrap(rssm_class.img_step)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    fn.decorator_list = []
    body, replacements = [], 0
    for node in fn.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "stoch"):
            require(ast.unparse(node.value) == "dist.sample(seed=nj.rng())", "Prior sample source changed")
            body += ast.parse("""
_audit_key = nj.rng()
stoch = jax.vmap(lambda row: jaxutils.OneHotDist(logits=row).sample(seed=_audit_key))(
    stats['logit'].astype(f32))
""").body
            replacements += 1
        elif isinstance(node, ast.Return):
            require(ast.unparse(node.value) == "cast(prior)", "Prior return source changed")
            body += ast.parse("""
_audit_result = cast(prior)
_audit_result['_audit_prior_key'] = _audit_key
return _audit_result
""").body
        else:
            body.append(node)
    require(replacements == 1, "Expected one prior categorical sample")
    fn.body = body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<common-key-native-RSSM-prior>", "exec"), namespace)
    return nj._scope_method(namespace["img_step"]), ast.unparse(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("SAFEDREAMER_CUDA_PATHS_READY") != "1":
        # Match the first-query process: glibc reads this path before imports.
        # Re-exec precedes output creation, the deadline and all JAX imports.
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
    source = (args.source or ROOT / protocol["output"] / "native_query_attempt1").resolve()
    out = (args.output or ROOT / protocol["output"] / "replay_attempt1").resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen_protocol.json").write_bytes(CONFIG.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    summary = {"status": "starting", "source": str(source.relative_to(ROOT)),
               "protocol": str(CONFIG.relative_to(ROOT)), "stage": "source_qualification",
               "activity": {"checkpoint_loads": 0, "replays_attempted": 0, "replays_completed": 0,
                            "rssm_batch_steps_completed": 0, "candidate_transitions_completed": 0,
                            "environment_resets": 0, "environment_steps": 0,
                            "encoder_calls": 0, "full_policy_calls": 0,
                            "model_api_calls": 0, "training_updates": 0},
               "checks": {}, "calibration_passed_before_common": False,
               "active_replay": None, "mode_gate_passed": None,
               "clean_competence_measured": False, "actual_harm_measured": False,
               "peak_process_gpu_mib": 0,
               "scope": "Fixed first-query final-round mode reachability; no physical control or task qualification."}
    stop = threading.Event()

    def persist():
        summary["elapsed_seconds"] = time.monotonic() - started
        save_json(out / "summary.json", summary)

    def check(label, ok):
        summary["checks"][label] = bool(ok)
        persist()
        require(ok, label)

    def deadline(*_):
        raise TimeoutError("Frozen final-round replay time or memory budget exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.setitimer(signal.ITIMER_REAL, protocol["resource_limits"]["final_round_replays_wall_seconds_max"])
    persist()
    try:
        previous = json.loads((source / "summary.json").read_text())
        frozen = json.loads((source / "frozen_protocol.json").read_text())
        check("qualified_native_query", previous["status"] == "completed_native_first_query_preflight"
              and not previous["native_final_round"]["selected_padding"])
        for field in ("source_revision", "checkpoint", "fixed_instance", "sampler_contract", "identification_controls"):
            check("frozen_" + field, frozen[field] == protocol[field])
        check("native_gpu_backend", previous["backend"] == "gpu")
        limits = protocol["resource_limits"]
        check("frozen_replay_budget", (limits["final_round_replays_max"], limits["replay_rssm_batch_steps_max"],
              limits["replay_candidate_transitions_max"]) == (2, 30, 15000))
        check("checkpoint_size", CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == protocol["checkpoint"]["bytes"])
        gpu = next(row for row in gpu_rows() if row["index"] == 0)
        summary["gpu_before"] = gpu
        if gpu["free_mib"] < limits["gpu_free_required_mib"]:
            summary["status"] = "resource_deferred"
            return 2
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        fraction = min(limits["gpu_allocator_budget_gib"] * 1024 / gpu["total_mib"], .20)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
        summary["allocator_fraction"] = fraction
        def monitor():
            while not stop.wait(.5):
                try:
                    used = own_gpu_mib()
                    summary["peak_process_gpu_mib"] = max(summary["peak_process_gpu_mib"], used)
                    if used > limits["gpu_memory_budget_gib"] * 1024:
                        summary["resource_limit_reason"] = "per_process_memory_budget_exceeded"
                        os.kill(os.getpid(), signal.SIGALRM)
                        return
                except Exception:
                    pass
        threading.Thread(target=monitor, daemon=True).start()
        def no_network(*_a, **_kw):
            raise RuntimeError("Network prohibited inside cached replay")
        socket.create_connection = no_network
        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        sys.path.insert(0, str(SOURCE))
        sys.path.insert(0, str(SOURCE / "SafeDreamer"))
        import numpy as np
        import jax
        import jax.numpy as jnp
        import embodied
        from SafeDreamer import agent as agt, behaviors, nets, ninjax as nj, jaxutils
        summary["versions"] = {key: importlib.metadata.version(key) for key in previous["versions"]}
        check("same_runtime", summary["versions"] == previous["versions"])
        jax.config.update("jax_platform_name", "gpu")
        jax.config.update("jax_transfer_guard", "disallow")
        jaxutils.COMPUTE_DTYPE = jnp.float16
        cfg = embodied.Config(agt.Agent.configs["defaults"]).update(agt.Agent.configs["osrp"])
        cfg = cfg.update({"jax.platform": "gpu", "seed": 0, "envs.parallel": "none", "envs.restart": False})
        check("native_parameters", (cfg.planner.horizon, cfg.planner.num_samples, cfg.planner.num_elites,
              cfg.planner.iterations, cfg.planner.mixture_coef, cfg.cost_limit,
              cfg.rssm.stoch, cfg.rssm.classes, cfg.rssm.unimix) == (15, 500, 50, 6, 0, 2, 48, 48, .01))
        with (source / "native_first_query.pkl").open("rb") as handle:
            cached = pickle.load(handle)
        last = cached["rounds"][-1]
        reference = last["trajectory"]
        check("candidate_shape", np.shape(last["actions"]) == (16, 500, 2))
        check("prior_keys", np.shape(reference["_audit_prior_key"]) == (16, 2)
              and reference["_audit_prior_key"].dtype == np.uint32)
        check("initial_metadata_sentinel", np.array_equal(reference["_audit_prior_key"][0], np.zeros(2, np.uint32)))
        posterior = {key: np.asarray(reference[key][0, :1]) for key in ("deter", "stoch", "logit")}
        check("replicated_posterior", all(np.array_equal(reference[key][0], np.repeat(value, 500, axis=0))
              for key, value in posterior.items()))
        check("same_policy_posterior", all(np.array_equal(cached["policy_state"][0][0][key], value)
              for key, value in posterior.items()))
        posterior["is_terminal"] = np.zeros(1, np.float32)
        require("observation_space" in previous and "action_space" in previous,
                "Native observation/action space metadata missing; do not guess from logging images")
        def spaces(items):
            return {key: embodied.Space(np.dtype(row["dtype"]), tuple(row["shape"])) for key, row in items.items()}
        obs_space, act_space = spaces(previous["observation_space"]), spaces(previous["action_space"])
        check("action_space", tuple(act_space["action"].shape) == (2,))
        with CHECKPOINT.open("rb") as handle:
            variables = pickle.load(handle)["agent"]
        summary["activity"]["checkpoint_loads"] = 1
        old_manifest = json.loads((source / "inference_consumed_keys.json").read_text())
        used = set()
        original_get = nj.Module.get
        def strict_get(self, key, *args, **kwargs):
            path = self.path + "/" + key
            if path in nj.context():
                require(path in old_manifest, "Replay requested an unqualified inference parameter: " + path)
                value = nj.context()[path]
                leaves, tree = jax.tree_util.tree_flatten(value)
                description = {"tree_structure": str(tree), "leaves": [
                    {"shape": list(v.shape), "dtype": str(v.dtype)} for v in leaves]}
                require(description == old_manifest[path], "Replay parameter schema differs: " + path)
                used.add(path)
            return original_get(self, key, *args, **kwargs)
        nj.Module.get = strict_get
        replay_method, replay_source = compile_replay(behaviors.CCEPlanner, nj)
        common_method, common_source = compile_common_prior(nets.RSSM, nj)
        (out / "native_final_round_extracted.py").write_text(replay_source + "\n")
        (out / "common_prior_transformation.py").write_text(common_source + "\n")
        install_audit_methods(agt, nets, behaviors, nj)
        native = agt.Agent.inner(obs_space, act_space, embodied.Counter(), cfg, name="agent")
        device = jax.devices("gpu")[0]
        variables = jax.device_put(variables, device)
        inputs = jax.device_put((posterior, last["mean_before"], last["std_before"],
                                 last["rng_before"], tuple(last["reserve_before"])), device)

        def make_run():
            def inner(latent, mean, std, key, reserve):
                return replay_method(native.expl_behavior, latent, mean, std, key, reserve)
            def run(v, args):
                return nj.pure(inner)(v, args[3], *args, create=False, modify=False)
            return jax.jit(run, device=device)

        def compare_tree(label, observed, expected):
            if isinstance(expected, dict):
                check(label + "_keys", set(observed) == set(expected))
                for key in expected:
                    compare_tree(label + "/" + key, observed[key], expected[key])
            else:
                a, b = np.asarray(observed), np.asarray(expected)
                check(label, a.shape == b.shape and a.dtype == b.dtype
                      and np.isfinite(a).all() and np.array_equal(a, b))

        def evaluate(label):
            summary["stage"] = label
            summary["active_replay"] = label
            summary["activity"]["replays_attempted"] += 1
            persist()
            tick = time.monotonic()
            output, _ = make_run()(variables, inputs)
            output = jax.device_get(output)
            summary["activity"]["replays_completed"] += 1
            summary["activity"]["rssm_batch_steps_completed"] += 15
            summary["activity"]["candidate_transitions_completed"] += 7500
            summary["active_replay"] = None
            summary.setdefault("replay_seconds", {})[label] = time.monotonic() - tick
            with (out / (label + ".pkl")).open("wb") as handle:
                pickle.dump(output, handle)
            persist()
            return output

        independent = evaluate("independent")
        compare_tree("independent_trajectory_exact", independent["trajectory"], reference)
        for key in ("actions", "costs", "returns", "safe_count", "elite_indices_in_native_filtered_array", "mean_after", "std_after"):
            compare_tree("independent_" + key + "_exact", independent[key], last[key])
        compare_tree("independent_returned_action_exact", independent["action"][None], cached["outputs"]["action"])
        summary["calibration_passed_before_common"] = True
        persist()
        nets.RSSM.img_step = common_method
        common = evaluate("common")
        compare_tree("common_actions_exact", common["actions"], last["actions"])
        compare_tree("common_prior_keys_exact", common["trajectory"]["_audit_prior_key"], reference["_audit_prior_key"])
        compare_tree("common_rng_after_exact", common["rng_after"], independent["rng_after"])
        check("common_reserve_after_exact", len(common["reserve_after"]) == len(independent["reserve_after"])
              and all(np.array_equal(a, b) for a, b in zip(common["reserve_after"], independent["reserve_after"])))
        for key in ("deter", "stoch", "logit"):
            compare_tree("common_initial_" + key + "_exact", common["trajectory"][key][0], reference[key][0])
        check("common_finite", all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(common)))
        summary["arms"] = {}
        for label, data in (("independent", independent), ("common", common)):
            count = int(data["safe_count"])
            indices = np.asarray(data["elite_indices_in_native_filtered_array"])
            padded = count >= 50 and bool(np.any(indices >= count))
            original_ids = np.flatnonzero(np.asarray(data["costs"]) < 2) if count >= 50 else np.arange(500)
            selected_ids = [int(original_ids[i]) if i < len(original_ids) else 0 for i in indices]
            summary["arms"][label] = {"safe_count": count, "mode": "minimum_cost" if count < 50 else "safe_reward",
                                     "selected_padding": padded, "elite_original_candidate_ids": selected_ids,
                                     "returned_action": np.asarray(data["action"]).tolist(),
                                     "always_minimum_cost_action": data["always_minimum_cost"]["action"].tolist(),
                                     "always_minimum_cost_elite_ids": data["always_minimum_cost"]["elite_original_candidate_ids"].tolist(),
                                     "mode_control_first_action_exact": np.array_equal(data["action"], data["always_minimum_cost"]["action"]),
                                     "mode_control_full_mean_exact": np.array_equal(data["mean_after"], data["always_minimum_cost"]["mean_after"])}
            if count < 50:
                for field in ("action", "mean_after", "std_after"):
                    compare_tree(label + "_already_minimum_cost_" + field,
                                 data["always_minimum_cost"][field], data[field])
        crossed = (summary["arms"]["independent"]["safe_count"] < 50) != (summary["arms"]["common"]["safe_count"] < 50)
        contaminated = any(arm["selected_padding"] for arm in summary["arms"].values())
        summary["mode_gate_passed"] = crossed and not contaminated
        summary["status"] = ("stopped_padding_contamination" if contaminated else
                             "completed_mode_switch_reached_clean_unqualified" if crossed else "completed_no_mode_switch_stop")
        summary["next_action"] = ("A separate unchanged native Clean qualification is required; no harm is measured."
                                  if summary["mode_gate_passed"] else "Stop this fixed operation; no query, root, threshold or coupling search.")
        summary["inference_consumed_keys"] = sorted(used)
        return 0
    except BaseException as exc:
        summary["status"] = ("resource_deferred" if isinstance(exc, TimeoutError) else
                             "calibration_failed" if isinstance(exc, CalibrationFailure) else "technical_error")
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary["partial_count_rule"] = "Completed counts exclude the active replay; its completed internal steps are unknown, not assumed zero."
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        persist()


if __name__ == "__main__":
    raise SystemExit(main())
