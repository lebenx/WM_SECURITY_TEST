#!/usr/bin/env python3
"""One native GenSafeNav case with passive forecast/feedback instrumentation.

No shadow calibration runs here. The original, including its known feedback
time-indexing defect, drives the actor. Terminal truth is captured inside the
physical environment before DummyVecEnv can reset it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "01baf926a5c77c1a4ab28635658eb014ef4f1767"
GST = ("gst_updated/results/100-gumbel_social_transformer-faster_lstm-"
       "lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run(args, out, summary):
    import importlib
    import numpy as np
    import torch

    native_load = torch.load
    def counted_load(*a, **kw):
        summary["activity"]["checkpoint_loads"] += 1
        summary.setdefault("checkpoint_load_paths", []).append(str(a[0] if a else kw.get("f")))
        result = native_load(*a, **kw)
        summary["activity"]["checkpoint_loads_completed"] += 1
        return result
    torch.load = counted_load

    vendor = args.vendor.resolve()
    os.chdir(vendor)
    sys.path.insert(0, str(vendor))
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy
    from rl.networks.network_utils import AddBias

    native_add_bias_init = AddBias.__init__
    if args.device == "cpu":
        def device_local_add_bias_init(self, bias, config):
            if not config.policy.constant_std:
                return native_add_bias_init(self, bias, config)
            # Native constant_std hard-codes .cuda() on this unregistered tensor,
            # so actor.to(cpu) cannot move it. Keep its values and non-Parameter
            # status exactly; allocate it on the requested inference backend.
            torch.nn.Module.__init__(self)
            self.config = config
            self._bias = bias.unsqueeze(1).to(config.training.device)
        AddBias.__init__ = device_local_add_bias_init
    summary["compatibility_adapters"] = [{
        "source": "rl/networks/network_utils.py:AddBias.__init__",
        "condition": "CPU and saved config.policy.constant_std",
        "change": "Unregistered constant bias uses configured device instead of hard-coded cuda; unchanged values, shape, dtype, non-Parameter status, forward and deterministic mean action.",
        "native_source_modified": False,
    }] if args.device == "cpu" else []

    # The saved get_args parses sys.argv; give it only the execution device flag.
    saved_argv = sys.argv
    sys.argv = ["test.py"] + (["--no-cuda"] if args.device == "cpu" else [])
    try:
        algo = importlib.import_module("trained_models.Ours_GST.arguments").get_args()
    finally:
        sys.argv = saved_argv
    config = importlib.import_module("trained_models.Ours_GST.configs.config").Config()
    assert algo.seed == 42 and algo.env_name == "CrowdSimPredRealGST-v0"
    assert config.pred.model_dir == GST and config.policy.aci_input
    assert config.env.use_wrapper and config.sim.predict_steps == 5
    assert config.sim.human_num == 20 and config.sim.human_num_range == 0
    assert config.env.time_step == 0.25 and config.env.time_limit == 50
    device = torch.device(args.device)
    assert bool(algo.cuda) == (device.type == "cuda"), "Requested device unavailable"
    # Native saved configuration hard-codes cuda:0 for GST independently of args.
    # Keep both components on the chosen backend; no numerical/model adaptation.
    config.training.device = str(device)
    for key in ("noise_clip_for_conformity_scores", "noise_std_for_conformity_scores",
                "noise_std_for_cost", "noise_clip_for_cost"):
        setattr(config.aci_related, key, 0.0)
    config.reward.base_collision_penalty = -20
    config.render_traj = config.save_slides = False
    config.save_path = str(out / "unused_render_output")
    config.args = algo
    torch.manual_seed(algo.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(algo.seed)
        torch.backends.cudnn.benchmark = not algo.cuda_deterministic
        torch.backends.cudnn.deterministic = algo.cuda_deterministic
    torch.set_num_threads(1)
    summary["runtime"] = {"python": sys.executable, "torch": torch.__version__,
                          "numpy": np.__version__, "device": str(device)}
    summary["scientific_settings"] = {
        "actor": "trained_models/Ours_GST/checkpoints/05207.pt",
        "gst": GST + "/checkpoint/epoch_100.pt", "seed": 42, "test_case": 0,
        "time_step": config.env.time_step, "time_limit": config.env.time_limit,
        "horizons": [1, 2, 3, 4, 5], "known_native_time_join_bug_preserved": True,
        "shadow_calibration_feeds_actor": False,
        "device_adaptation": "Only saved GST device made consistent with actor backend"}
    summary["weights"] = {p: (vendor / p).stat().st_size for p in
                          (summary["scientific_settings"]["actor"],
                           summary["scientific_settings"]["gst"])}

    activity = summary["activity"]
    envs = make_vec_envs(algo.env_name, algo.seed, 1, algo.gamma, str(out), device,
                         allow_early_resets=True, config=config, ax=None,
                         test_case=0, pretext_wrapper=True)
    base = envs.venv.unwrapped.envs[0].env
    assert base.case_counter["test"] == 0 and base.phase == "test"
    # Native make_env only honors test_case when a render axis is supplied.
    base.set_test_case(0)
    summary["initial_case_selection"] = {"test_case": base.test_case,
                                         "case_counter": dict(base.case_counter),
                                         "thisSeed": int(base.thisSeed),
                                         "explicit_headless_case_selection": True}
    summary["scientific_settings"]["pred_interval"] = int(envs.pred_interval)
    assert envs.pred_interval == 1
    actor = Policy(envs.observation_space.spaces, envs.action_space, config,
                   base_kwargs=algo, base=config.robot.policy)
    actor.load_state_dict(torch.load(vendor / summary["scientific_settings"]["actor"],
                                    map_location=device), strict=True)
    actor.base.nenv = 1
    # Native test.py does not call actor.eval(); retain that exact mode.
    torch.nn.DataParallel(actor).to(device)
    summary["strict_actor_load_passed"] = True
    summary["actor_training_mode"] = actor.training

    entity_objects, entity_ids = [], {}
    reset_count = 0
    terminal_seen = False
    latest_prediction = None
    files = {name: (out / (name + ".jsonl")).open("w")
             for name in ("issues", "physical_steps", "resets")}

    def entity(h):
        key = id(h)
        if key not in entity_ids:
            entity_ids[key] = len(entity_objects)
            entity_objects.append(h)  # Prevent Python object-ID reuse.
        return entity_ids[key]

    def emit(name, row):
        files[name].write(json.dumps(row, allow_nan=False) + "\n")
        files[name].flush()

    def snapshot():
        rows = []
        visible = getattr(base, "human_visibility", [])
        for index, h in enumerate(base.humans):
            rows.append({"lifetime_id": entity(h), "human_id": int(h.id),
                         "position": [float(h.px), float(h.py)],
                         "velocity": [float(h.vx), float(h.vy)],
                         "goal": [float(h.gx), float(h.gy)],
                         "visibility_observation": bool(visible[index]),
                         "issue_valid": bool(h.prediction_valid),
                         "native_update_counts": [len(a.true_values) for a in h.pred_error_aci_list],
                         "native_last_residuals": [float(a.true_values[-1]) if a.true_values else None
                                                   for a in h.pred_error_aci_list]})
        return {"reset_index": reset_count, "physical_index": int(base.step_counter),
                "time": float(base.global_time), "humans": rows,
                "robot_position": [float(base.robot.px), float(base.robot.py)],
                "robot_goal": [float(base.robot.gx), float(base.robot.gy)]}

    def deadline():
        if time.monotonic() - summary["_started"] > args.wall_seconds:
            raise TimeoutError("Single-case wall budget exceeded; no retry")

    native_reset, native_step = base.reset, base.step
    native_predict, native_talk = envs.predictor.forward, base.talk2Env
    robot_class = type(base.robot)
    robot_step_was_own = "step" in robot_class.__dict__
    native_robot_step = robot_class.step
    applied_controls = []

    def observed_robot_step(robot, action):
        # Class-level transparent forwarding also remains correct for deepcopy'd
        # simulator robots; only the live physical robot contributes a record.
        result = native_robot_step(robot, action)
        if robot is base.robot:
            applied_controls.append({"type": type(action).__name__,
                                     "values": list(map(float, action))})
            activity["actual_robot_controls"] += 1
        return result

    def observed_reset(*a, **kw):
        nonlocal reset_count
        activity["physical_resets"] += 1
        result = native_reset(*a, **kw)
        reset_count += 1
        emit("resets", {"snapshot": snapshot(), "native_env_seed": int(base.rand_seed),
                        "post_terminal_autoreset": terminal_seen})
        deadline()
        return result

    def observed_step(*a, **kw):
        nonlocal terminal_seen
        deadline()
        assert not terminal_seen, "Second physical episode is forbidden"
        before = snapshot()
        applied_controls.clear()
        activity["physical_steps"] += 1
        result = native_step(*a, **kw)
        activity["physical_steps_completed"] += 1
        _, reward, done, info = result
        after = snapshot()  # Before DummyVecEnv's automatic reset.
        terminal_seen = bool(done)
        action = np.asarray(a[0] if a else kw["action"]).tolist()
        assert len(applied_controls) == 1, "Expected one live robot control per physical step"
        emit("physical_steps", {"before": before, "after": after,
                                "nominal_actor_action": action,
                                "actual_robot_control": dict(applied_controls[0]),
                                "reward": float(reward), "done": bool(done),
                                "info_type": type(info["info"]).__name__,
                                "cost": float(info["cost"])})
        deadline()
        return result

    def observed_predict(*a, **kw):
        nonlocal latest_prediction
        deadline()
        activity["gst_calls"] += 1
        result = native_predict(*a, **kw)
        activity["gst_calls_completed"] += 1
        trajectory, mask = result
        input_traj = kw["input_traj"] if "input_traj" in kw else a[0]
        input_mask = kw["input_binary_mask"] if "input_binary_mask" in kw else a[1]
        arrays = {"input_traj": input_traj.detach().cpu().numpy().copy(),
                  "input_binary_mask": input_mask.detach().cpu().numpy().copy(),
                  "output_traj": trajectory.detach().cpu().numpy().copy(),
                  "out_mask": mask.detach().cpu().numpy().copy()}
        index = activity["gst_calls"]
        path = out / "predictor" / f"call_{index:04d}.npz"
        np.savez(path, **arrays)
        latest_prediction = {"path": str(path.relative_to(out)), "arrays": arrays,
                             "call": index, "post_terminal": terminal_seen}
        deadline()
        return result

    def observed_talk(*a, **kw):
        activity["talk2env_calls"] += 1
        result = native_talk(*a, **kw)
        assert latest_prediction is not None
        if terminal_seen:
            activity["post_terminal_talk2env_calls"] += 1
            return result  # Native evaluator does this; never admit as case-0 data.
        arrays = latest_prediction["arrays"]
        raw_mask = arrays["out_mask"][0, :, 0].astype(bool)
        input_mask = arrays["input_binary_mask"][0, :, :, 0].astype(bool)
        humans = sorted(base.humans, key=lambda h: np.linalg.norm(
            [base.robot.px - h.px, base.robot.py - h.py]))
        data = np.asarray(a[0] if a else kw["data"])
        rows = []
        for row, h in enumerate(humans):
            actual = data[row].reshape(6, 2) + [base.robot.px, base.robot.py]
            hid = int(h.id)
            raw_future = arrays["output_traj"][0, hid, :, :2]
            # The wrapper subtracts its float32 observed robot position before
            # sorting. Reproduce that operation, not a loose world-space match.
            observed_robot = np.asarray([base.robot.px, base.robot.py], dtype=raw_future.dtype)
            expected_relative = raw_future - observed_robot
            delivered_relative = data[row].reshape(6, 2)[1:]
            exact_gst_match = bool(np.array_equal(expected_relative, delivered_relative))
            rows.append({"lifetime_id": entity(h), "human_id": hid,
                         "sorted_row": row, "issue_valid": bool(h.prediction_valid),
                         "gst_out_mask": bool(raw_mask[hid]),
                         "visible": bool(input_mask[hid, -1]),
                         "previous_sample_visible": bool(input_mask[hid, -2]),
                         "gst_delivered_relative_exact": exact_gst_match,
                         "gst_delivered_relative_max_abs": float(np.max(np.abs(
                             expected_relative - delivered_relative))),
                         "forecast_world_xy": actual.tolist(),
                         "native_interval": (h.last_aci_predicted_conformity_score.tolist()
                                             if h.last_aci_predicted_conformity_score is not None else None)})
        emit("issues", {"physical_index": int(base.step_counter),
                        "time": float(base.global_time), "reset_index": reset_count,
                        "gst_call": latest_prediction["call"],
                        "predictor_cache": latest_prediction["path"], "humans": rows})
        return result

    base.reset, base.step = observed_reset, observed_step
    robot_class.step = observed_robot_step
    envs.predictor.forward, base.talk2Env = observed_predict, observed_talk
    (out / "predictor").mkdir()
    try:
        # Matches the first native evaluate episode, including terminal autoreset.
        base.episode_k = 0
        obs = envs.reset()
        assert base.rand_seed == 1042 and reset_count == 1
        def receive_prediction(current_obs):
            values, _ = base.talk2Env(current_obs["spatial_edges"].cpu().numpy()[0])
            current_obs["conformity_scores"] = torch.from_numpy(
                np.array([values])).to(torch.float32).to(device)
        receive_prediction(obs)
        hidden = {"human_node_rnn": torch.zeros(1, 1, actor.base.human_node_rnn_size, device=device),
                  "human_human_edge_rnn": torch.zeros(1, actor.base.human_num + 1,
                                                      actor.base.human_human_edge_rnn_size, device=device)}
        masks = torch.zeros(1, 1, device=device)
        limit = math.ceil(base.time_limit / base.time_step) + 2
        total_reward = 0.0
        for index in range(limit):
            deadline()
            activity["actor_calls"] += 1
            with torch.no_grad():
                _, action, _, hidden = actor.act(obs, hidden, masks, deterministic=True)
            activity["actor_calls_completed"] += 1
            deadline()
            obs, reward, done, infos = envs.step(action)
            receive_prediction(obs)
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            if (index + 1) % 20 == 0 or bool(done[0]):
                print(json.dumps({"steps": index + 1, "return": total_reward,
                                  "done": bool(done[0]), "activity": activity}), flush=True)
                write_json(out / "progress.json", {"steps": index + 1, "activity": activity})
            masks = torch.tensor([[0.0] if d else [1.0] for d in done], dtype=torch.float32, device=device)
            if bool(done[0]):
                deadline()
                name = type(infos[0]["info"]).__name__
                summary.update(status="completed_single_native_clean_episode",
                               outcome={"terminal": name, "success": name == "ReachGoal",
                                        "collision": name == "Collision", "timeout": name == "Timeout",
                                        "return": total_reward, "steps": index + 1})
                break
        else:
            raise RuntimeError("Native episode did not terminate within native time budget +2")
    finally:
        for handle in files.values():
            handle.close()
        envs.close()
        torch.load = native_load
        AddBias.__init__ = native_add_bias_init
        if robot_step_was_own:
            robot_class.step = native_robot_step
        else:
            del robot_class.step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "vendor/GenSafeNav")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gensafenav_feedback_selection_gate_20261015.json")
    parser.add_argument("--resources", type=Path, default=ROOT / "runs/gensafenav_resources_20261015/provenance.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/gensafenav_feedback_selection_gate_20261015")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"status": "starting", "_started": started, "activity": {
        "actor_calls": 0, "gst_calls": 0, "physical_steps": 0, "physical_resets": 0,
        "actor_calls_completed": 0, "gst_calls_completed": 0, "physical_steps_completed": 0,
        "checkpoint_loads": 0, "checkpoint_loads_completed": 0, "actual_robot_controls": 0,
        "talk2env_calls": 0, "post_terminal_talk2env_calls": 0},
        "scope": "One native case-0 Clean trace; passive feedback records, no attack or shadow intervention"}
    exit_code = 0
    try:
        assert 0 < args.wall_seconds <= 600
        raw_manifest = args.resources.resolve().read_bytes()
        manifest = json.loads(raw_manifest)
        assert manifest["status"] == "completed_git_blob_verified_resource_fetch"
        assert manifest["revision"] == REVISION
        assert Path(manifest["destination"]).resolve() == args.vendor.resolve()
        for record in manifest["files"]:
            path = args.vendor.resolve() / record["path"]
            assert path.stat().st_size == record["bytes"] and record["git_blob_verified"]
            digest = hashlib.sha1(f'blob {record["bytes"]}\0'.encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == record["git_blob_sha1"], record["path"]
        summary["revision"] = REVISION
        summary["resource_manifest"] = {"path": str(args.resources.resolve()),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(), "files_verified": len(manifest["files"])}
        shutil.copy2(__file__, out / "launched_script.py")
        shutil.copy2(args.config.resolve(), out / "frozen_protocol.json")
        protocol = json.loads(args.config.resolve().read_text())
        assert protocol["source_revision"] == REVISION
        assert protocol["fixed_run"]["test_case"] == 0
        assert protocol["fixed_run"]["model_environment_wall_seconds"] == args.wall_seconds
        run(args, out, summary)
    except Exception as error:
        exit_code = 1
        summary.update(status="technical_failure_retained", error=repr(error), traceback=traceback.format_exc())
        write_json(out / "activity_failed.json", summary["activity"])
        print(summary["traceback"], flush=True)
    finally:
        summary.pop("_started", None)
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
