#!/usr/bin/env python3
"""Preregistered CPU prerequisites for released JEPA-WM on MetaWorld reach-wall.

No model forward, CUDA, training, remote dataset or API is used here. A possible
later native GPU stage has its scientific protocol fixed here but needs its own
registered execution adapter after these prerequisites pass.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/pilot_dino_metaworld_clean_gate.py"
SERVO_HELPER = ROOT / "scripts/pilot_dino_metaworld_servo_baseline.py"
OFFICIAL = ROOT / "vendor/jepa-wms/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
MODEL = ROOT / "models/JEPA_WM/mw_jepa-wm.pth.tar"
MUJOCO_TARGET = ROOT / ".deps/dino_mujoco330"
MUJOCO_LIBRARY = MUJOCO_TARGET / "mujoco/libmujoco.so.3.3.0"
WALL_POLICY = ROOT / ".venv-a2world/lib/python3.11/site-packages/metaworld/policies/sawyer_reach_wall_v3_policy.py"
RESOURCE_AUDIT = ROOT / "reports/jepa_metaworld_reach_wall_resource_audit_20261015.json"
DEFAULT_CONFIG = ROOT / "configs/jepa_metaworld_reach_wall_cpu_v1.json"
DEFAULT_OUT = ROOT / "runs/jepa_metaworld_reach_wall_cpu_v1"
SEEDS = [1, 2, 3]
BRANCHES = ["noop_repeat0", "noop_repeat1", "straight_servo", "wall_aware_servo"]
BOUNDARIES = [1, 16, 31, 46, 61, 76, 91]
HOLDS = [15] * 6 + [9]
NORMALIZE = [[.485, .456, .406], [.229, .224, .225]]
PROTOCOL = {
    "task": "mw-reach-wall", "seeds": SEEDS, "branch_order_per_seed": BRANCHES,
    "attempts_per_branch": 1, "total_environment_steps": 100,
    "warmup_zero_controls": 1, "policy_control_budget": 99,
    "query_elapsed_steps": BOUNDARIES, "hold_control_counts": HOLDS,
    "physical_seed_rule": "Native rank0/local_seed1 ep0,1,2 gives seeds1,2,3; fresh environment per branch, global setup seed1",
    "goal": "Unmodified native separate-expert final visual/proprio after100 controls; expert action sequence is archived only",
    "noop": "Physical float32 [0,0,0,0], not normalized zero; two complete99-control repeats",
    "servo_inputs": "Only raw float32 current and goal proprio of shape(4); xyz used, fourth element ignored",
    "straight_servo": "Frozen reach comparator: clip(float32(5)*(goal_xyz-current_xyz),-1,1), gripper0; collision-unaware, not official wall expert",
    "wall_aware_servo": "Same P5/gripper0 with public desired-position rule: if -.1<=x<=.3 and .60<=y<=.80 and z<.25, desired=goal_xyz+[0,0,1], else desired=goal_xyz",
    "wall_aware_prior": "Uses public fixed wall geometry prior; uses supplied expert-goal proprio instead of hidden environment target. Not full official expert: feedback is latched at7 boundaries and goal is observed endpoint",
    "servo_arithmetic": "float32; clip xyz to[-1,1] before environment; fixed action for15x6+9 controls",
    "privileged_policy_inputs": False,
    "evaluation_only_fields": ["task target", "TCP", "full state", "expert actions", "reward", "success"],
    "primary_success": "Final native info.success at environment step100; TCP-target distance<=.05; done_at_succ=False",
    "secondary": ["ever_success", "final_reach_distance", "reward sum excluding warmup"],
    "renderer": "CPU OSMesa", "mujoco": "3.3.0 isolated target", "threads": 4,
    "cpu_model_loading": "Instantiate released checkpoint and local DINO encoder on CPU; force strict=True for every load_state_dict; no forward",
    "model_forward_calls": 0, "api_requests": 0, "training_updates": 0,
    "future_native_gpu": {
        "checkpoint": "JEPA_WM/mw_jepa-wm.pth.tar", "actual_image_size": 224,
        "image_normalization": NORMALIZE, "predictor": "AdaLN", "proprio_cost_retained": True,
        "cem": {"samples": 300, "iterations": 15, "elites": 10, "horizon": 6,
                "num_act_stepped": 3, "alpha": .1, "context": 2},
        "precision": "FP32, native full300 candidate batch; no autocast/microbatch or retraining",
        "macro_action": "20D = five4D controls; frameskip5/action_skip1; native last planned10 truncated to9",
        "cem_rng": "One native GC_Agent generator seeded1 across fixed seeds1,2,3 in order",
        "heads_cfg": {}, "decode_each_iteration": False,
        "execution": "Not implemented/launched by this CPU prerequisites script; any later adapter must bind this registration and all CPU artifacts",
    },
    "failure_rule": "No retries, seed replacement, gain/goal/threshold tuning or longer budget. Preserve each scheduled branch and error. Prerequisite errors block GPU; all fixed CPU branches may still be collected once without selection.",
}
GATE = {
    "complete_fixed_three_seeds_required": True,
    "all_initial_states_nontrivial": True, "all_expert_final_success": True,
    "all_exact_initial_goal_setup_across_four_branches": True,
    "all_noop_full_trace_repeat_exact": True, "strict_checkpoint_load_required": True,
    "minimum_native_final_successes": 2,
    "minimum_native_minus_wall_aware_final_successes": 2,
    "minimum_native_minus_noop_final_successes": 2,
    "straight_servo_is_descriptive_comparator": True,
    "cpu_mathematical_stop": "If wall-aware or noop has>=2/3 final successes, native<=3/3 cannot clear the net+2 gate; stop panel GPU with native unmeasured, not failed",
    "runtime_or_pairing_failure": "Incomplete/invalid prerequisites, never counted as native task failure",
    "scope": "Fixed three-seed descriptive normal-capability/resource screen, not statistical confirmation, pure forecast causality, attack or defense evidence",
    "stop_rule": "No new task/seed selection or budget changes to obtain a passing panel; all prior failed gates remain unchanged",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg_for(helper, out, device="cpu"):
    import yaml
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(yaml.safe_load(OFFICIAL.read_text()))
    h = helper.require
    h((cfg.planner.iterations, cfg.planner.num_samples, cfg.planner.num_elites,
       cfg.planner.horizon, cfg.planner.num_act_stepped) == (15, 300, 10, 6, 3), "Official CEM changed")
    h(cfg.task_specification.task == "mw-reach-wall" and cfg.task_specification.max_episode_steps == 100
      and cfg.task_specification.goal_source == "expert" and cfg.task_specification.succ_def == "simu"
      and not cfg.task_specification.done_at_succ and cfg.task_specification.num_frames == 1
      and cfg.task_specification.num_proprios == 1 and cfg.task_specification.obs == "rgb_state", "Official task changed")
    h(cfg.model_kwargs.data.img_size == cfg.task_specification.img_size == 224
      and [list(v) for v in cfg.model_kwargs.data_aug.normalize] == NORMALIZE, "Official JEPA image contract changed")
    h(cfg.model_kwargs.data.custom.frameskip == 5 and cfg.model_kwargs.data.custom.action_skip == 1
      and cfg.model_kwargs.data.custom.normalize_action and cfg.model_kwargs.wrapper_kwargs.ctxt_window == 2
      and cfg.planner.planning_objective.alpha == .1 and not cfg.planner.planning_objective.sum_all_diffs,
      "Official temporal/objective contract changed")
    mk = cfg.model_kwargs.pretrain_kwargs
    h(mk.predictor.pred_type == "AdaLN" and mk.predictor.pred_depth == 6
      and mk.predictor.pred_embed_dim == 384 and mk.predictor.pred_num_heads == 16
      and mk.action_conditioning == "token" and mk.action_encoder.action_encoder_inpred
      and mk.action_encoder.action_tokens == 1 and mk.proprio_encoding == "feature"
      and mk.proprio_encoder.proprio_emb_dim == 16 and not mk.proprio_encoder.proprio_encoder_inpred
      and mk.visual_encoder.enc_version == "dinov2_vits14", "Released JEPA architecture changed")
    cfg.work_dir, cfg.device = str(out), device
    cfg.rank, cfg.world_size, cfg.local_seed = 0, 1, 1
    cfg.frameskip, cfg.action_ratio, cfg.action_dim = 5, 5, 4
    cfg.tasks = ["mw-reach-wall"]
    cfg.task_specification.multitask = False
    cfg.logging.optional_plots = False
    cfg.logging.tqdm_silent = True
    cfg.planner.decode_each_iteration = False
    cfg.planner.distribute_planner = False
    cfg.model_kwargs.pretrain_kwargs.heads_cfg = {}
    return cfg


def wall_aware_action(servo, current, goal):
    """Only the two observed proprio inputs enter the public geometry rule."""
    import numpy as np
    # Validate even when the geometry condition is false or fourth entries unused.
    servo.servo_action(current, goal)
    desired = goal.copy()
    x, y, z = current[:3]
    if -.1 <= x <= .3 and .60 <= y <= .80 and z < .25:
        desired[:3] = desired[:3] + np.asarray([0., 0., 1.], dtype=np.float32)
    return servo.servo_action(current, desired)


def setup_checks(left, right):
    return {"all_setup_arrays_exact": left["arrays"] == right["arrays"],
            **{key + "_bytes_exact": left[key]["sha256"] == right[key]["sha256"]
               for key in ("initial_image", "goal_image", "expert_trace")}}


def runtime(helper):
    import mujoco
    import torch
    helper.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CUDA must remain unavailable/uninitialized")
    helper.require(os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "Renderer differs")
    helper.require(str(MUJOCO_TARGET) in os.environ.get("PYTHONPATH", "").split(os.pathsep), "Isolated MuJoCo target missing")
    helper.require(Path(mujoco.__file__).resolve().is_relative_to(MUJOCO_TARGET)
                   and mujoco.__version__ == mujoco.mj_versionString() == "3.3.0", "MuJoCo runtime differs")
    return {"mujoco_import": str(Path(mujoco.__file__).resolve()), "mujoco_engine": mujoco.mj_versionString(),
            "cuda_initialized": False, "environment_created": False, "model_forward_calls": 0}


def inventory(helper):
    with patch.object(helper, "OFFICIAL", OFFICIAL):
        sources = helper.source_inventory()
    for path in (Path(__file__).resolve(), SERVO_HELPER, WALL_POLICY, WALL_POLICY.with_name("policy.py"), MUJOCO_LIBRARY):
        sources[str(path)] = helper.sha(path)
    return sources


def prepare(helper, config_path, out):
    helper.require(not config_path.exists() and not out.exists(), "Registration/output already exists")
    cfg_for(helper, out)
    runtime_info = runtime(helper)
    config = {
        "schema_version": 1, "status": "registered_before_any_reach_wall_model_or_environment_execution",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": helper.bind(__file__), "frozen_helper": helper.bind(HELPER), "servo_helper": helper.bind(SERVO_HELPER),
        "official_config": helper.bind(OFFICIAL), "model": helper.bind(MODEL),
        "dino_encoder": helper.bind(helper.DINO_WEIGHTS), "wall_policy_source": helper.bind(WALL_POLICY),
        "resource_audit": helper.bind(RESOURCE_AUDIT), "out_dir": str(out.resolve()),
        "protocol": PROTOCOL, "gate": GATE, "versions": helper.versions(), "runtime": runtime_info,
        "source_sha256": inventory(helper),
        "mujoco_installation": helper.bind(MUJOCO_TARGET / "installation_provenance.json"),
        "planned_branches": [{"seed": seed, "condition": branch} for seed in SEEDS for branch in BRANCHES],
        "adaptations": [
            "New independent cfg_for uses official JEPA architecture,224/ImageNet; filename r256 does not override YAML224.",
            "Import frozen DINO helper in an isolated module and temporarily replace only its MODEL variable with JEPA checkpoint for its strict CPU loader; restore in finally via context manager. Helper cfg_for/run are not used.",
            "Reuse frozen native environment/expert setup,trace,outcome; fresh environment per branch, global seed1 then native seeded_rand_vec/task/randvec flow. This is not unchanged official48-episode reuse.",
            "Disable optional visualization heads/decoding/plots. No latent planning component is replaced.",
            "Frozen setup helper blocks only unused eager training-dataset init_data import and raises if called. Actual native expert/env/model/CEM modules and DATA_STATS stay unchanged.",
            "Same frozen raw-proprio P5 comparator, plus public wall-rule goal-proprio adaptation with7 feedback points. Full official expert actions are never supplied to either controller or future planner.",
            "CPU OSMesa and isolated MuJoCo3.3.0; existing shared virtual environments unchanged.",
        ],
        "limitations": [
            "Python3.11/Torch2.6/torchvision.21 differ from upstream Python<3.11/Torch>=2.7/torchvision>=.22. No claim that these differences cause outcomes or that this is exact official reproduction.",
            "Installed MetaWorld3.1.1 requires MuJoCo3.3.0; isolated runtime explicitly satisfies that physics dependency.",
            "Official separate expert goal generation has100 actions and no main warmup; it verifies goal reachability but is not a same-initial-state99-control expert performance baseline.",
            "Recorded physics/proprio/image equality covers named arrays and observations, not every hidden controller/render state.",
            "Same-seed normal task screen; wall name alone does not establish difficulty or need for a learned world model. Training task coverage is not established.",
            "All old attack and capability records remain unchanged. No intervention or security-success claim.",
        ],
        "run_command": "PYTHONPATH=" + str(MUJOCO_TARGET) + " .venv-jepa-native/bin/python scripts/pilot_jepa_metaworld_reach_wall.py --cpu-check --config " + str(config_path.resolve()),
    }
    helper.write(config_path, config, exclusive=True)
    return helper.bind(config_path)


def preflight(helper, config_path):
    config, digest = helper.read(config_path)
    helper.require(config["protocol"] == PROTOCOL and config["gate"] == GATE, "Registered protocol changed")
    helper.require(config["planned_branches"] == [{"seed": s, "condition": b} for s in SEEDS for b in BRANCHES], "Schedule changed")
    for key, path in (("runner", __file__), ("frozen_helper", HELPER), ("servo_helper", SERVO_HELPER),
                      ("official_config", OFFICIAL), ("model", MODEL), ("dino_encoder", helper.DINO_WEIGHTS),
                      ("wall_policy_source", WALL_POLICY), ("resource_audit", RESOURCE_AUDIT),
                      ("mujoco_installation", MUJOCO_TARGET / "installation_provenance.json")):
        helper.require(config[key] == helper.bind(path), "Changed binding: " + key)
    helper.require(config["versions"] == helper.versions(), "Installed package versions differ")
    helper.require(config["source_sha256"] == inventory(helper), "Source/environment inventory differs")
    helper.require(config["runtime"] == runtime(helper), "Runtime differs")
    cfg_for(helper, Path(config["out_dir"]))
    return config, {"status": "passed_cpu_preflight", "config": {"path": str(config_path.resolve()), "sha256": digest},
                    "source_files": len(config["source_sha256"]), "model_forward_calls": 0,
                    "model_instances_created": 0, "environment_steps": 0, "cuda_initialized": False}


def run_servo(helper, servo, env, initial, goal, directory, condition):
    import numpy as np
    import torch
    current = initial["proprio"][-1].detach().cpu().numpy().copy()
    target = goal["proprio"][-1].detach().cpu().numpy().copy()
    policy = servo.servo_action if condition == "straight_servo" else lambda c, g: wall_aware_action(servo, c, g)
    rows, queries = [], []
    with (directory / "trace.jsonl").open("x") as handle:
        for index, (elapsed, count) in enumerate(zip(BOUNDARIES, HOLDS), 1):
            helper.require(env.elapsed_steps() == elapsed, "Query boundary differs")
            action = policy(current, target)
            query = {"index": index, "elapsed_before": elapsed, "current_proprio": current.tolist(),
                     "goal_proprio": target.tolist(), "action": action.tolist(), "executed_control_count": 0}
            queries.append(query)
            for j in range(count):
                _, info, row = helper.trace_step(env, torch.from_numpy(action.copy()), index, j)
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                query["executed_control_count"] += 1
                helper.require(row["done"] == (len(rows) == 99), "Unexpected termination")
            current = info["proprio"][-1].detach().cpu().numpy().copy()
            helper.write(directory / "decisions.json", queries)
    # Independently reconstruct each policy input from the actual preceding trace.
    with np.load(directory / "setup.npz", allow_pickle=False) as arrays:
        initial_state = arrays["initial_state"].copy()
    offset = 0
    for q, count in zip(queries, HOLDS):
        state = initial_state if offset == 0 else rows[offset - 1]["state"]
        actual_current = np.asarray(state[:4], dtype=np.float32)
        expected = policy(actual_current, target).tolist()
        helper.require(q["current_proprio"] == actual_current.tolist() and q["goal_proprio"] == target.tolist()
                       and q["action"] == expected, "Controller input/action mismatch")
        helper.require(all(r["action"] == expected and r["planning_call"] == q["index"]
                           and r["action_in_chunk"] == j for j, r in enumerate(rows[offset:offset + count])), "Held action mismatch")
        offset += count
    return {"outcome": helper.outcome(rows, directory), "decisions": helper.bind(directory / "decisions.json"),
            "decision_audit": {"all99_actions_from_recorded_raw_proprio": True, "decision_count": 7}}


def execute_branch(helper, servo, env, initial, goal, directory, condition):
    """Retain the last already-produced observation, without extra env queries."""
    import numpy as np
    from PIL import Image
    latest = {}
    original_step = env.step
    def observe(action, *args, **kwargs):
        result = original_step(action, *args, **kwargs)
        latest["pixels"] = result[0].detach().cpu().numpy().copy()
        latest["proprio"] = result[4]["proprio"].detach().cpu().numpy().copy()
        latest["state"] = np.asarray(result[4]["state"]).copy()
        return result
    try:
        env.step = observe
        result = ({"outcome": helper.noop(env, directory)} if condition.startswith("noop")
                  else run_servo(helper, servo, env, initial, goal, directory, condition))
    finally:
        env.step = original_step
    final = {**helper.physical(env), **latest}
    np.savez(directory / "final.npz", **final)
    Image.fromarray(latest["pixels"][-1].transpose(1, 2, 0)).save(directory / "final.png")
    result["final_observation"] = {"arrays": {k: helper.array_digest(v) for k, v in final.items()},
                                   "npz": helper.bind(directory / "final.npz"),
                                   "image": helper.bind(directory / "final.png"),
                                   "additional_environment_steps_or_renders": 0}
    return result


def decision(episodes, strict_load_ok):
    complete = len(episodes) == 3 and all(e["status"] == "completed" for e in episodes)
    valid = complete and strict_load_ok and all(e["prerequisites_passed"] for e in episodes)
    counts = None
    if complete:
        counts = {name: sum(e["branches"][BRANCHES.index(name)]["outcome"]["final_success"] for e in episodes)
                  for name in ("noop_repeat0", "straight_servo", "wall_aware_servo")}
    stop = valid and (counts["noop_repeat0"] >= 2 or counts["wall_aware_servo"] >= 2)
    return {"cpu_prerequisites_passed": valid, "complete_panel": complete, "final_success_counts": counts,
            "gpu_progression_allowed": bool(valid and not stop),
            "decision": "incomplete_or_invalid_prerequisites" if not valid else
                        "stop_panel_native_unmeasured_mathematically_cannot_pass" if stop else "eligible_for_separately_registered_native_gpu_stage",
            "native_results_measured": False, "native_success_count": None, "native_gate_passed": None,
            "scope": GATE["scope"]}


def cpu_check(helper, servo, config_path):
    import torch
    config, validation = preflight(helper, config_path)
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = cfg_for(helper, out)
    report = {"status": "running", "config": validation["config"], "runner": helper.bind(__file__),
              "protocol": PROTOCOL, "gate": GATE, "episodes": [], "planned_branches": config["planned_branches"],
              "strict_model_load": None, "decision": None, "model_forward_calls": 0,
              "cuda_initialized": False, "api_requests": 0, "training_updates": 0}
    helper.write(out / "summary.json", report)
    try:
        def reject_forward(*args, **kwargs):
            raise RuntimeError("All torch module forwards forbidden in CPU prerequisites")
        with patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden in CPU prerequisites")), \
             patch.object(torch.nn.Module, "_call_impl", reject_forward):
            with patch.object(helper, "MODEL", MODEL):
                model, _, loading = helper.load_model(cfg, "cpu")
            report["strict_model_load"] = loading
            helper.require(loading["strict_loads"] and all(not c["missing"] and not c["unexpected"] for c in loading["strict_loads"]), "Strict loading incomplete")
            helper.require(not torch.cuda.is_initialized(), "Strict loader initialized CUDA")
            del model
            helper.write(out / "summary.json", report)
            for seed in SEEDS:
                episode = {"seed": seed, "status": "running", "branches": [], "prerequisites_passed": False}
                report["episodes"].append(episode)
                for condition in BRANCHES:
                    branch = {"condition": condition, "status": "preparing"}
                    episode["branches"].append(branch)
                    env = None
                    directory = out / f"seed{seed}_{condition}"
                    try:
                        env, initial, goal, setup = helper.setup_episode(cfg, seed, directory)
                        branch["setup"] = setup
                        branch.update(execute_branch(helper, servo, env, initial, goal, directory, condition))
                        branch["status"] = "completed"
                    except Exception as error:
                        branch.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
                    finally:
                        if env is not None:
                            env.close()
                        helper.write(out / "summary.json", report)
                    print(json.dumps({"seed": seed, "condition": condition, "status": branch["status"],
                                      "final_success": branch.get("outcome", {}).get("final_success")}), flush=True)
                if all(b["status"] == "completed" for b in episode["branches"]):
                    reference = episode["branches"][0]
                    checks = {b["condition"]: setup_checks(reference["setup"], b["setup"]) for b in episode["branches"][1:]}
                    checks["noop_repeat_trace"] = {"bytes_exact": reference["outcome"]["trace"]["sha256"] == episode["branches"][1]["outcome"]["trace"]["sha256"]}
                    checks["noop_repeat_final"] = {
                        "arrays_exact": reference["final_observation"]["arrays"] == episode["branches"][1]["final_observation"]["arrays"],
                        "png_exact": reference["final_observation"]["image"]["sha256"] == episode["branches"][1]["final_observation"]["image"]["sha256"],
                    }
                    checks["task_prerequisites"] = {
                        "all_nontrivial": all(not b["setup"]["initial_success"] for b in episode["branches"]),
                        "all_expert_final_success": all(b["setup"]["expert_final_success"] for b in episode["branches"]),
                    }
                    episode.update(status="completed", checks=checks, prerequisites_passed=all(all(c.values()) for c in checks.values()))
                else:
                    episode["status"] = "error"
                helper.write(out / "summary.json", report)
        report["decision"] = decision(report["episodes"], report["strict_model_load"] is not None)
        report["status"] = "completed_cpu_validation" if report["decision"]["complete_panel"] else "incomplete_cpu_validation"
    except BaseException as error:
        report.update(status="error", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["cuda_initialized"] = torch.cuda.is_initialized()
        helper.write(out / "summary.json", report)


def self_test(helper, servo):
    import numpy as np
    tests = []
    cfg = cfg_for(helper, DEFAULT_OUT)
    helper.require(cfg.task_specification.img_size == 224 and not cfg.model_kwargs.pretrain_kwargs.heads_cfg, "Config adaptation incorrect")
    tests.append("actual224_ImageNet_JEPA_architecture_CEM_and_no_decoder")
    helper.require(sum(HOLDS) == 99 and BOUNDARIES == [1 + sum(HOLDS[:i]) for i in range(7)], "Schedule incorrect")
    tests.append("seven_feedback_boundaries_and99_controls")
    goal = np.asarray([.02, .88, .12, .8], dtype=np.float32)
    current = np.asarray([0, .7, .1, -.5], dtype=np.float32)
    helper.require(wall_aware_action(servo, current, goal)[2] == 1 and servo.servo_action(current, goal)[2] < 1, "Wall rule inactive")
    high = current.copy(); high[2] = .3
    helper.require(np.array_equal(wall_aware_action(servo, high, goal), servo.servo_action(high, goal)), "Wall rule outside region")
    tests.append("wall_rule_activates_only_in_registered_geometry")
    # Execute only the published pure desired-position function, not its package
    # imports or any environment. Compare the port at interior and boundary cases.
    tree = ast.parse(WALL_POLICY.read_text())
    official = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_desired_pos")
    official.decorator_list, official.returns = [], None
    for arg in official.args.args:
        arg.annotation = None
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[official], type_ignores=[])), str(WALL_POLICY), "exec"), namespace)
    for xyz in ([0, .7, .1], [0, .7, .25], [-.11, .7, .1], [.31, .7, .1],
                [0, .59, .1], [0, .81, .1], [-.1, .6, .1], [.3, .8, .1]):
        observed = np.asarray([*xyz, 0], dtype=np.float32)
        desired = namespace["_desired_pos"]({"hand_pos": observed[:3], "goal_pos": goal[:3]})
        expected_goal = goal.copy(); expected_goal[:3] = desired
        helper.require(np.array_equal(wall_aware_action(servo, observed, goal), servo.servo_action(observed, expected_goal)), "Public wall rule port differs")
    tests.append("published_pure_wall_rule_exact_on_eight_geometry_cases")
    c2, g2 = current.copy(), goal.copy(); c2[3], g2[3] = 99, -99
    helper.require(np.array_equal(wall_aware_action(servo, current, goal), wall_aware_action(servo, c2, g2)), "Ignored proprio input affects action")
    tests.append("fourth_proprio_ignored_gripper_zero")
    for malformed in (current.astype(np.float64), current[:3], np.asarray([0, np.nan, 0, 0], dtype=np.float32)):
        try:
            wall_aware_action(servo, malformed, goal)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed proprio accepted")
    tests.append("wrong_dtype_shape_and_nonfinite_inputs_rejected")
    fake = [{"status": "completed", "prerequisites_passed": True,
             "branches": [{"outcome": {"final_success": i < (2 if b == "wall_aware_servo" else 0)}} for b in BRANCHES]}
            for i in range(3)]
    result = decision(fake, True)
    helper.require(result["decision"].startswith("stop_panel") and result["native_success_count"] is None, "Mathematical stop changed")
    tests.append("strong_baseline_two_successes_stops_without_native_failure_claim")
    helper.require(not decision(fake[:2], True)["gpu_progression_allowed"] and decision(fake[:2], True)["final_success_counts"] is None, "Partial panel yields verdict")
    fake[0]["prerequisites_passed"] = False
    helper.require(not decision(fake, True)["gpu_progression_allowed"], "Invalid prerequisites allow GPU")
    tests.append("partial_and_invalid_prerequisites_block_GPU")
    original = helper.MODEL
    try:
        with patch.object(helper, "MODEL", MODEL):
            helper.require(helper.MODEL == MODEL, "Model adapter failed")
            raise RuntimeError("test restoration")
    except RuntimeError:
        pass
    helper.require(helper.MODEL == original, "Model binding not restored")
    tests.append("isolated_helper_checkpoint_adapter_restores_after_exception")
    return {"status": "passed", "tests": tests, "count": len(tests), "runner": helper.bind(__file__),
            "model_instances_created": 0, "environment_steps": 0, "model_forward_calls": 0, "cuda_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-registration", "preflight", "self-test", "cpu-check"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    helper = load_module("reach_wall_frozen_dino_helper", HELPER)
    servo = load_module("reach_wall_frozen_servo_helper", SERVO_HELPER)
    helper.bootstrap(cpu=True)
    if args.prepare_registration:
        result = prepare(helper, args.config.resolve(), args.out_dir.resolve())
    elif args.preflight:
        result = preflight(helper, args.config.resolve())[1]
    elif args.self_test:
        result = self_test(helper, servo)
    else:
        cpu_check(helper, servo, args.config.resolve())
        return
    if args.report:
        helper.write(args.report, result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
