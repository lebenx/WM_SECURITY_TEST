#!/usr/bin/env python3
"""CPU-only paired goal-marker rendering on a cached DINO-WM reach state.

No policy, encoder, world-model inference, training, or network call. Different
raw pixels do not establish that a latent-only forecast service can decode goals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".deps/dino_mujoco330"
OSMESA = ROOT / ".deps/osmesa/usr/lib/x86_64-linux-gnu"
UPSTREAM = ROOT / "vendor/jepa-wms"
SOURCE = ROOT / "runs/dino_metaworld_mujoco330_clean_v1/seed1_clean"
DEFAULT_OUT = ROOT / "runs/dino_goal_observability_20261015"
GOALS = {"goal_a": [-0.08, 0.84, 0.12], "goal_b": [0.08, 0.86, 0.24]}
SIM_FIELDS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def bootstrap():
    if os.environ.get("DINO_OBSERVABILITY_BOOTSTRAPPED") != "1":
        os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="osmesa",
                          PYOPENGL_PLATFORM="osmesa", LIBGL_ALWAYS_SOFTWARE="1",
                          DINO_OBSERVABILITY_BOOTSTRAPPED="1")
        os.environ["LD_LIBRARY_PATH"] = str(OSMESA) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["PYTHONPATH"] = str(RUNTIME) + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be disabled")
    require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "OSMesa required")
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(RUNTIME))


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": binding(__file__), "source_setup": binding(SOURCE / "setup.npz"),
        "source_image": binding(SOURCE / "initial.png"),
        "protocol": {
            "fixed_goals_before_rendering": GOALS, "source_seed": 1,
            "operation": "Restore cached physical fields; change only target and its native visible goal site; render original/A/B/original without any policy step",
            "goal_legality": "Native goal_space and planar object-goal distance >=0.15",
            "restore_mismatch_rule": "If historical pixels do not match, label the reconstructed state explicitly; never claim exact archived observation reproduction",
            "failure_rule": "Record failure and stop; no installation, dependency repair, alternate renderer or retries",
            "service_scope": "New hypothetical client/service split, not an existing deployment",
        },
        "activity": {"model_loads": 0, "encoder_calls": 0, "world_model_calls": 0,
                     "policy_calls": 0, "explicit_environment_step_calls": 0,
                     "explicit_environment_reset_calls": 0, "training_updates": 0,
                     "api_requests": 0, "dependency_installs": 0,
                     "constructor_reset_note": "Native goal-observable environment construction may reset/simulate internally; all paired measurements follow cached-state restoration and use render/forward only"},
        "input_boundary": {
            "actual_raw_model_observation": "visual RGB plus four-dimensional proprio=state[:4]; full 39D environment state is not supplied",
            "actual_rollout_interface": "model.unroll(z_ctxt, act_suffix): encoded visual/proprio context plus proposed action suffix; no goal argument",
            "coordinate_metadata": "Native goal-observable state[-3:] exposes target locally; this metadata is not automatically available to a forecast service",
            "goal_encoder": "A private-goal split would also need local goal encoding; sending goal image to a remote encoder directly reveals it",
        },
        "native_mean_query_source_fact": {
            "path": "vendor/jepa-wms/evals/simu_env_planning/planning/planning/planner.py",
            "lines": [303, 323, 333],
            "finding": "Each of 15 CEM iterations calls unroll on the 300-candidate batch and then on its updated mean; the last mean query contains the returned first three macro actions. This is source inspection, not a measured query transcript.",
            "privacy_limit": "Do not count the selected-action prefix in that mean query as information beyond the selected actions themselves",
        },
        "limits": [
            "One fixed development physical state and two preselected goals, not independent samples or a task-capability test",
            "Raw-pixel dependence only: no latent features, goal decoding, classifier accuracy, conditional leakage, attack success or task harm measured",
            "A target is a native visible site and reward parameter; this probe does not hide or remove it",
        ],
    }
    write(out / "summary.json", summary)
    env = None
    try:
        import numpy as np
        import torch
        import mujoco
        from PIL import Image
        from pilot_dino_metaworld_clean_gate import cfg_for
        from evals.simu_env_planning.envs.init import make_env

        torch.set_num_threads(4)
        require(mujoco.__version__ == "3.3.0" and str(RUNTIME) in str(Path(mujoco.__file__).resolve()), "Expected existing MuJoCo 3.3.0 overlay")
        require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
        summary["runtime"] = {"python": sys.version, "interpreter": sys.executable,
                              "mujoco": mujoco.__version__, "mujoco_module": mujoco.__file__,
                              "renderer": "OSMesa", "torch": torch.__version__}
        with np.load(SOURCE / "setup.npz", allow_pickle=False) as archive:
            cached = {key: archive[key].copy() for key in archive.files if not key.startswith("expert_") and not key.startswith("goal_")}
        cfg = cfg_for(out, "cpu")
        env = make_env(cfg)
        raw = env.proprio_env.unwrapped
        for name in SIM_FIELDS:
            getattr(raw.data, name)[:] = cached["sim_" + name]
        raw.data.time = float(cached["sim_time"])
        raw.model.site_pos[:] = cached["model_site_pos"]
        raw._target_pos = cached["target"].copy()
        raw._last_rand_vec = cached["rand_vec"].copy()
        raw._prev_obs = cached["previous_observation"].copy()
        mujoco.mj_forward(raw.model, raw.data)
        # mj_forward recomputes derived fields; retain the cached integration state.
        raw.data.qacc_warmstart[:] = cached["sim_qacc_warmstart"]
        goal_site = raw.model.site("goal").id
        original_target = raw._target_pos.copy()
        original_sites = raw.model.site_pos.copy()
        goal_rgba = raw.model.site_rgba[goal_site].copy()
        summary["marker"] = {"site_id": int(goal_site), "rgba": goal_rgba.tolist(),
                             "size": raw.model.site_size[goal_site].tolist(), "visibility_modified": False}

        def state_without_mutation():
            previous = raw._prev_obs.copy()
            state = raw._get_obs().astype(np.float32)
            raw._prev_obs = previous
            return state

        def physical():
            values = {name: getattr(raw.data, name).copy() for name in SIM_FIELDS}
            values.update(time=np.asarray(raw.data.time), object_position=raw.get_body_com("obj").copy(),
                          tcp=raw.tcp_center.copy(), previous_observation=raw._prev_obs.copy(),
                          current_proprio=state_without_mutation()[:4].copy())
            return values

        baseline = physical()
        summary["cached_integrator_fields_exact"] = {
            name: bool(np.array_equal(baseline[name], cached["sim_" + name])) for name in SIM_FIELDS}
        require(all(summary["cached_integrator_fields_exact"].values()), "Cached physical fields not restored exactly")
        summary["goal_legality"] = {}
        for label, xyz in GOALS.items():
            goal = np.asarray(xyz, dtype=np.float64)
            distance = float(np.linalg.norm(cached["rand_vec"][:2] - goal[:2]))
            legal = bool(raw.goal_space.contains(goal) and distance >= .15)
            summary["goal_legality"][label] = {"goal": xyz, "in_native_goal_space": bool(raw.goal_space.contains(goal)),
                                               "planar_distance_from_source_object": distance, "passed": legal}
            require(legal, "Preselected target is not legal: " + label)
        write(out / "summary.json", summary)

        frames, records, arrays = {}, {}, {}
        for label, xyz in [("original", original_target), *GOALS.items(), ("restored", original_target)]:
            goal = np.asarray(xyz, dtype=np.float64)
            raw._target_pos = goal.copy()
            raw.model.site_pos[:] = original_sites
            raw.model.site_pos[goal_site] = goal
            mujoco.mj_forward(raw.model, raw.data)
            raw.data.qacc_warmstart[:] = baseline["qacc_warmstart"]
            before = physical()
            pixels = env.proprio_env.render().copy()
            after = physical()
            require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8, "Unexpected native image schema")
            checks = {name: bool(np.array_equal(before[name], baseline[name]) and np.array_equal(after[name], baseline[name]))
                      for name in baseline}
            require(all(checks.values()), "Physical state drift during target rendering: " + label)
            require(np.array_equal(raw.model.site_rgba[goal_site], goal_rgba), "Marker appearance changed")
            non_goal_sites = np.arange(len(original_sites)) != goal_site
            require(np.array_equal(raw.model.site_pos[non_goal_sites], original_sites[non_goal_sites]), "Non-goal site changed")
            state = state_without_mutation()
            require(np.array_equal(state[-3:], goal.astype(np.float32)), "Native state does not expose expected target")
            path = out / (label + ".png")
            Image.fromarray(pixels).save(path)
            frames[label] = pixels
            records[label] = {"goal": goal.tolist(), "physical_exact_to_baseline_before_and_after_render": checks,
                              "state_first4_actual_proprio": state[:4].tolist(), "state_last3_local_target": state[-3:].tolist(),
                              "pixel_artifact": binding(path)}
            arrays[label + "_pixels"] = pixels
            arrays[label + "_state"] = state
            arrays.update({label + "_" + name: value for name, value in after.items()})
        old_pixels = cached["initial_pixels"][-1].transpose(1, 2, 0)
        old_png = np.asarray(Image.open(SOURCE / "initial.png").convert("RGB"))
        require(np.array_equal(old_pixels, old_png), "Historical PNG and saved initial tensor disagree")
        historical_exact = bool(np.array_equal(frames["original"], old_pixels))
        summary["state_origin"] = {
            "mode": "cached_physics_restored_exact_historical_pixels" if historical_exact else "cached_integration_state_restored_with_recomputed_derived_render",
            "historical_initial_pixels_exact": historical_exact,
            "historical_initial_proprio_exact": bool(np.array_equal(records["original"]["state_first4_actual_proprio"], cached["initial_proprio"][-1])),
            "historical_pixel_max_absolute_difference": int(np.abs(frames["original"].astype(np.int16) - old_pixels.astype(np.int16)).max()),
            "note": "If pixels differ, the render is a reconstructed cached integration state, not an asserted exact historical observation",
        }
        delta = np.abs(frames["goal_a"].astype(np.int16) - frames["goal_b"].astype(np.int16))
        changed = np.any(delta != 0, axis=-1)
        positions = np.argwhere(changed)
        summary["paired_pixel_comparison"] = {
            "changed_pixels": int(changed.sum()), "total_pixels": int(changed.size),
            "changed_pixel_fraction": float(changed.mean()), "changed_channel_values": int((delta != 0).sum()),
            "max_absolute_channel_difference": int(delta.max()), "mean_absolute_channel_difference": float(delta.mean()),
            "changed_pixel_bbox_yx_inclusive": [positions.min(0).tolist(), positions.max(0).tolist()] if len(positions) else None,
            "raw_pixels_depend_on_target": bool(changed.any()),
        }
        summary["restoration"] = {"original_pixels_exact": bool(np.array_equal(frames["original"], frames["restored"])),
                                  "all_model_sites_exact": bool(np.array_equal(raw.model.site_pos, original_sites)),
                                  "target_exact": bool(np.array_equal(raw._target_pos, original_target))}
        require(all(summary["restoration"].values()), "Original target restoration failed")
        summary["renders"] = records
        np.savez_compressed(out / "paired_observations.npz", **arrays)
        summary["paired_observations"] = binding(out / "paired_observations.npz")
        summary["status"] = "completed_cpu_paired_goal_render_gate"
        summary["decision"] = {
            "raw_pixel_goal_blindness": "rejected_for_this_pair" if changed.any() else "no_pixel_difference_observed_for_this_pair",
            "latent_only_service_goal_privacy": "unmeasured_not_rejected_by_pixel_difference",
            "native_query_leakage_attack": "not_tested",
            "task_capability_for_new_goals": "not_tested",
        }
        summary["cuda_initialized"] = torch.cuda.is_initialized()
        require(not summary["cuda_initialized"], "CUDA unexpectedly initialized")
    except BaseException as error:
        summary.update(status="failed_closed", exception_type=type(error).__name__,
                       exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "state_origin": summary["state_origin"],
                      "paired_pixel_comparison": summary["paired_pixel_comparison"], "decision": summary["decision"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    bootstrap()
    main(args.out)
