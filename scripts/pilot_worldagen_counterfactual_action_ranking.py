#!/usr/bin/env python3
"""Counterfactual action-ranking test for the WorldAgen goal bus.

WorldAgen's released controller emits actions directly from its policy head;
the native rollout does not use the world forecast for selection.  This script
therefore evaluates an explicitly labeled, independent forecast-driven
planner.  It freezes one candidate action set generated once from the clean
policy, scores every candidate with CLIP similarity between the predicted
future image and the original task text, and then replays only the selected
candidate in a fresh LIBERO environment.  The replay reward/success is the
actual utility; no future observation is used for selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import clip
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "LIBERO"))
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_worldagen_goal_bus_replicate as probe  # noqa: E402
import pilot_worldagen_libero_closed_loop as libero  # noqa: E402
from worldagen_forecast_layout import primary_forecasts

OUT = Path(os.environ.get(
    "WORLDAGEN_RANKING_OUT",
    str(ROOT / "runs/worldagen_counterfactual_action_ranking_v0/summary.json"),
))
DEVICE = probe.DEVICE


def batch_from_observation(model, obs: dict, task_text: str, candidates: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Build the two-chunk WorldAgen input with duplicated current context."""
    primary_one = model.image_processor(
        Image.fromarray(obs["agentview_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180)
    )
    wrist_one = model.image_processor(
        Image.fromarray(obs["robot0_eye_in_hand_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180)
    )
    primary = torch.stack([primary_one, primary_one]).unsqueeze(0).to(DEVICE)
    wrist = torch.stack([wrist_one, wrist_one]).unsqueeze(0).to(DEVICE)
    ori = libero.quaternion_to_euler(obs["robot0_eef_quat"])
    gripper = float(obs["robot0_gripper_qpos"][0]) if np.asarray(obs["robot0_gripper_qpos"]).size else -1.0
    state_one = torch.tensor(
        [*np.asarray(obs["robot0_eef_pos"], dtype=np.float32), *ori, gripper],
        dtype=torch.float32,
    )
    state = state_one.repeat(2, 1).unsqueeze(0).to(DEVICE)
    text = clip.tokenize([task_text]).unsqueeze(1).repeat(candidates.shape[0], 2, 1).to(DEVICE)
    primary = primary.repeat(candidates.shape[0], 1, 1, 1, 1)
    wrist = wrist.repeat(candidates.shape[0], 1, 1, 1, 1)
    state = state.repeat(candidates.shape[0], 1, 1)
    actions = candidates.to(DEVICE)
    return primary, wrist, state, text, actions


def patchify(images: torch.Tensor) -> torch.Tensor:
    n, c, h, w = images.shape
    p = 16
    x = images.reshape(n, c, h // p, p, w // p, p)
    x = torch.einsum("nchpwq->nhwpqc", x).reshape(n, (h // p) * (w // p), p * p * c)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1e-6).sqrt()


def unpatch_for_clip(tokens: torch.Tensor, reference_processed: torch.Tensor | None = None) -> Image.Image:
    """Turn normalized predicted patches into a deterministic pseudo-image.

    WorldAgen's released image target is patch-normalized.  We use a global
    affine display transform only for the fixed CLIP ranking score; forecast
    error is computed separately in patch space against the true replay image.
    """
    token_grid = tokens.detach().float().reshape(14, 14, 16, 16, 3)
    if reference_processed is not None:
        # Use only current-observation patch statistics to invert the released
        # per-patch normalization. No future frame or candidate outcome enters.
        ref = reference_processed.detach().float().reshape(3, 224, 224)
        ref_grid = ref.reshape(3, 14, 16, 14, 16).permute(1, 3, 2, 4, 0)
        mean = ref_grid.mean(dim=(2, 3), keepdim=True)
        std = ref_grid.var(dim=(2, 3), keepdim=True).add(1e-6).sqrt()
        norm_grid = token_grid * std + mean
        norm = norm_grid.permute(0, 2, 1, 3, 4).reshape(224, 224, 3)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=norm.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=norm.device)
        x = (norm * clip_std + clip_mean).clamp(0, 1)
        x = (x * 255.0).byte().cpu().numpy()
    else:
        x = token_grid.permute(0, 2, 1, 3, 4).reshape(224, 224, 3)
        lo, hi = float(x.min()), float(x.max())
        x = ((x - lo) / max(hi - lo, 1e-6) * 255.0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(x)


@torch.inference_mode()
def rank_candidates(model, obs: dict, model_goal_text: str, score_goal_text: str,
                    candidates: torch.Tensor, mask: torch.Tensor,
                    clip_reconstruction: str = "global_minmax") -> tuple[np.ndarray, torch.Tensor]:
    batch = batch_from_observation(model, obs, model_goal_text, candidates)
    world, _policy = probe.infer(model, batch, mask)
    # model.py:481 returns [batch * image-time, view, patches, pixels].
    n = candidates.shape[0]
    world = primary_forecasts(world, n)
    # The score target is frozen to the original task goal in every condition.
    text_token = clip.tokenize([score_goal_text]).to(DEVICE)
    text_feature = model.clip_model.encode_text(text_token).float()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference_processed = None
    if clip_reconstruction == "reference_patch":
        reference_processed = model.image_processor(
            Image.fromarray(obs["agentview_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180)
        ).to(DEVICE)
    scores = []
    for i in range(candidates.shape[0]):
        image = model.image_processor(unpatch_for_clip(world[i, 0], reference_processed)).unsqueeze(0).to(DEVICE)
        image_feature = model.clip_model.encode_image(image).float()
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores.append(float((image_feature * text_feature).sum().item()))
    return np.asarray(scores, dtype=np.float64), world.detach().cpu()


def make_env(task, suite_name: str, seed: int, init_index: int = 0):
    from libero.libero.envs import OffScreenRenderEnv
    bddl = libero.LIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224, render_gpu_device_id=0)
    env.task_id = 0; env.task_name = task.name; env.task_suite_name = suite_name
    env.seed(seed); env.reset()
    init_file = libero.LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    init_states = torch.load(init_file, map_location="cpu", weights_only=False)
    if not 0 <= int(init_index) < len(init_states):
        raise IndexError(f"init_index={init_index} outside {len(init_states)} states for task={task.name}")
    obs = env.set_init_state(init_states[int(init_index)])
    for _ in range(5):
        obs, *_ = env.step(np.zeros(7, dtype=np.float32))
    return env, obs


def _state_position(env, name: str) -> np.ndarray:
    state = env.env.object_states_dict[name]
    return np.asarray(state.get_geom_state()["pos"], dtype=np.float64)


def _articulation_progress(env, name: str, predicate: str) -> float | None:
    state = env.env.object_states_dict[name]
    parent_name = getattr(state, "parent_name", name)
    if parent_name not in env.env.object_states_dict:
        return None
    parent_state = env.env.object_states_dict[parent_name]
    q = parent_state.get_joint_state()
    if not q:
        return None
    obj = env.env.get_object(parent_name)
    ranges = obj.object_properties.get("articulation", {}).get(
        {"open": "default_open_ranges", "close": "default_close_ranges",
         "turnon": "default_turnon_ranges", "turnoff": "default_turnoff_ranges"}.get(predicate, ""),
        [],
    )
    if not ranges:
        return None
    lo, hi = float(min(ranges)), float(max(ranges))
    qpos = float(np.asarray(q).reshape(-1)[0])
    # Zero at the target interval and negative outside it. This is a true
    # simulator joint-coordinate progress measure, not a model score.
    return 0.0 if lo <= qpos <= hi else -min(abs(qpos - lo), abs(qpos - hi))


def true_task_progress(env) -> float:
    """Compute a simulator-only, goal-directed progress score.

    The score is intentionally independent of WorldAgen and CLIP. Articulation
    goals use distance to the BDDL joint interval; binary on/in goals use the
    Euclidean distance between the actual object and target-site geometry.
    A success predicate receives score 0, while unfinished goals are negative.
    """
    values = []
    for goal in env.env.parsed_problem["goal_state"]:
        predicate = str(goal[0]).lower()
        if predicate in {"open", "close", "turnon", "turnoff"}:
            value = _articulation_progress(env, goal[1], predicate)
            if value is not None:
                values.append(value)
                continue
        if len(goal) >= 3 and goal[1] in env.env.object_states_dict and goal[2] in env.env.object_states_dict:
            distance = float(np.linalg.norm(_state_position(env, goal[1]) - _state_position(env, goal[2])))
            values.append(-distance)
        else:
            values.append(0.0 if env.check_success() else -1.0)
    return float(np.mean(values)) if values else (0.0 if env.check_success() else -1.0)


def eef_approach_distance(env, obs: dict) -> float:
    """Distance from the real end effector to the first goal entity."""
    goals = env.env.parsed_problem["goal_state"]
    if not goals:
        return float("nan")
    goal = goals[0]
    target_name = goal[1] if len(goal) >= 2 else None
    if target_name not in env.env.object_states_dict:
        return float("nan")
    target = _state_position(env, target_name)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return float(np.linalg.norm(eef - target))


def replay_once(env, obs: dict, actions: torch.Tensor) -> dict:
    """Replay one fixed action trajectory in one fresh environment."""
    del obs  # the environment already holds the reset state
    initial_progress = true_task_progress(env)
    # Obtain the initial rendered state from the live simulator, avoiding any
    # model-derived state. This quantity is a secondary approach utility.
    initial_obs = env.env._get_observations()
    initial_eef_distance = eef_approach_distance(env, initial_obs)
    rewards = []
    success = False
    observations = {}
    last_obs = initial_obs
    action_sequence = actions.detach().cpu().numpy()
    for action in action_sequence:
        _obs, reward, done, info = env.step(action.astype(np.float32).tolist())
        last_obs = _obs
        rewards.append(float(reward))
        if len(rewards) in (3, 6):
            observations[len(rewards)] = _obs
        success = bool(done) or bool(info.get("success", False)) if isinstance(info, dict) else bool(done)
        if success:
            break
    try:
        checked = bool(env.check_success())
    except Exception:
        checked = success
    final_progress = true_task_progress(env)
    final_eef_distance = eef_approach_distance(env, last_obs)
    return {
        "cumulative_reward": float(sum(rewards)),
        "max_reward": float(max(rewards) if rewards else 0.0),
        "success": bool(success or checked),
        "environment_steps": len(rewards),
        "observations_after_steps": observations,
        "true_progress_initial": initial_progress,
        "true_progress_final": final_progress,
        "true_progress_delta": final_progress - initial_progress,
        "eef_approach_distance_initial": initial_eef_distance,
        "eef_approach_distance_final": final_eef_distance,
        "eef_approach_progress": initial_eef_distance - final_eef_distance,
    }


def replay_repeated(task, suite_name: str, seed: int, actions: torch.Tensor,
                    repeats: int = 1) -> dict:
    """Run independent fresh-environment replays and aggregate their metrics.

    The first replay's observations are retained for forecast-error scoring.
    For repeats > 1, the returned top-level numeric metrics are means and the
    individual episodes are preserved under ``repeat_results``.
    """
    episode_results = []
    for _ in range(max(1, int(repeats))):
        replay_env, replay_obs = make_env(task, suite_name, seed)
        try:
            episode_results.append(replay_once(replay_env, replay_obs, actions))
        finally:
            replay_env.close()
    if len(episode_results) == 1:
        return episode_results[0]

    result = copy.deepcopy(episode_results[0])
    # Keep image observations only at the top level (the first replay); the
    # nested repeat records are metrics-only so the summary remains JSON-safe.
    compact_results = []
    for episode in episode_results:
        compact = {key: value for key, value in episode.items()
                   if key != "observations_after_steps"}
        compact_results.append(compact)
    result["repeat_results"] = compact_results
    result["replay_count"] = len(episode_results)
    result["success_rate"] = float(np.mean([bool(x["success"]) for x in episode_results]))
    result["success"] = bool(any(bool(x["success"]) for x in episode_results))
    for key in (
        "cumulative_reward", "max_reward", "environment_steps",
        "true_progress_initial", "true_progress_final", "true_progress_delta",
        "eef_approach_distance_initial", "eef_approach_distance_final",
        "eef_approach_progress",
    ):
        result[key] = float(np.mean([float(x[key]) for x in episode_results]))
    # Forecast error is computed against the first replay's observations;
    # ranking never uses any replay observation.
    return result


def flip_rate(reference: np.ndarray, other: np.ndarray) -> float:
    flips = total = 0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            a = np.sign(reference[i] - reference[j])
            b = np.sign(other[i] - other[j])
            if a == 0 or b == 0:
                continue
            total += 1
            flips += int(a != b)
    return float(flips / total) if total else 0.0


def forecast_mse(world: torch.Tensor, obs_after: list[dict], model) -> float:
    targets = []
    for obs in obs_after:
        primary = model.image_processor(Image.fromarray(obs["agentview_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180))
        targets.append(patchify(primary.unsqueeze(0))[0])
    target = torch.stack(targets)
    # Only the first chunk has temporally matched current context. The second
    # input duplicates the current frame, so it is not a valid t+6 forecast.
    pred = world[:1].float().cpu()
    if pred.shape[0] != target.shape[0]:
        pred = pred[:target.shape[0]]
    return float((pred - target[:pred.shape[0]]).square().mean().item())


def recorded_candidates(root: Path, task_index: int, episode: int | None,
                        count: int, horizon: int) -> torch.Tensor:
    """Load fixed action chunks from held-out real LIBERO demonstrations."""
    rows = []
    for file in sorted((root / "data/chunk-000").glob("file-*.parquet")):
        rows.extend(pq.read_table(file, columns=["episode_index", "task_index", "frame_index", "action"]).to_pylist())
    rows = [r for r in rows if int(r["task_index"]) == task_index and (episode is None or int(r["episode_index"]) == episode)]
    rows.sort(key=lambda r: (int(r["episode_index"]), int(r["frame_index"])))
    if len(rows) < horizon:
        raise RuntimeError(f"not enough recorded actions for task={task_index}, episode={episode}")
    starts = np.linspace(0, len(rows) - horizon, count, dtype=int)
    return torch.tensor(
        [[[float(x) for x in rows[start + j]["action"]] for j in range(horizon)] for start in starts],
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--replay-repeats", type=int, default=1,
                        help="independent fresh-environment repeats of the selected fixed action chunk; selection remains one chunk")
    parser.add_argument("--recorded-replay-horizon", type=int, default=30,
                        help="length of recorded trajectory executed after ranking; only its first --horizon actions are scored")
    parser.add_argument("--candidate-source", choices=["clean_policy", "recorded"], default="clean_policy")
    parser.add_argument("--candidate-data-root", default=str(ROOT / "data/worldagen_goal_probe_extra"))
    parser.add_argument("--candidate-task-index", type=int, default=None)
    parser.add_argument("--candidate-episode", type=int, default=None)
    parser.add_argument("--candidate-map", default=None,
                        help="recorded mapping task_id:data_task_index:data_episode, comma separated")
    parser.add_argument("--goal-offsets", default="1",
                        help="legal task offsets to evaluate, comma separated; default is the adjacent Goal-Swap")
    parser.add_argument("--evaluate-all-candidates", action="store_true",
                        help="replay every frozen candidate once and report true-utility regret for selected actions")
    parser.add_argument("--clip-reconstruction", choices=["global_minmax", "reference_patch"], default="global_minmax")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the WorldAgen ranking test")
    from libero.libero import benchmark
    model = probe.build_model()
    native, bus_cut = libero.masks(model)
    control_cut = native.clone()
    policy_rows = torch.tensor([37, 38, 39, 77, 78, 79, 77, 78, 79])
    policy_cols = torch.tensor([0, 0, 0, 0, 0, 0, 40, 40, 40])
    control_cut[policy_rows, policy_cols] = -float("inf")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.task_ids.split(",") if x]
    goal_offsets = [int(x) for x in args.goal_offsets.split(",") if x]
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    all_rows = []
    candidate_map = {}
    if args.candidate_map:
        for item in args.candidate_map.split(","):
            tid, did, ep = item.split(":")
            candidate_map[int(tid)] = (int(did), int(ep))
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env, obs = make_env(task, args.suite, args.seed)
        clean_goal = task.language
        num_tasks = suite.get_num_tasks() if hasattr(suite, "get_num_tasks") else len(suite.tasks)
        goal_texts = {offset: suite.get_task((task_id + offset) % num_tasks).language for offset in goal_offsets}
        # Candidate set is generated once and then frozen across all conditions.
        if args.candidate_source == "recorded":
            mapped = candidate_map.get(task_id)
            candidate_task_index = mapped[0] if mapped else args.candidate_task_index
            candidate_episode = mapped[1] if mapped else args.candidate_episode
            if candidate_task_index is None:
                raise SystemExit("--candidate-task-index is required with --candidate-source recorded")
            candidates = recorded_candidates(Path(args.candidate_data_root), candidate_task_index,
                                              candidate_episode, args.candidates, args.recorded_replay_horizon)
        else:
            zero = torch.zeros(1, 6, 7, device=DEVICE)
            clean_batch = batch_from_observation(model, obs, clean_goal, zero)
            _w, clean_policy = probe.infer(model, clean_batch, native)
            base_action = clean_policy[0].reshape(-1, 7).detach().cpu()[:args.horizon]
            if base_action.shape[0] < args.horizon:
                base_action = torch.cat([base_action, base_action[-1:].repeat(args.horizon - base_action.shape[0], 1)], dim=0)
            candidates = [base_action.clamp(-1, 1), torch.zeros_like(base_action)]
            for _ in range(max(0, args.candidates - len(candidates))):
                noise = torch.randn(base_action.shape, generator=rng) * 0.15
                candidates.append((base_action + noise).clamp(-1, 1))
            candidates = torch.stack(candidates[:args.candidates])
        # Score only the fixed first action chunk. Recorded candidates may
        # carry a longer open-loop continuation for the independent replay.
        input_actions = candidates[:, :args.horizon]
        if input_actions.shape[1] < 6:
            input_actions = torch.cat([input_actions, input_actions[:, -1:].repeat(1, 6 - input_actions.shape[1], 1)], dim=1)
        conditions = {"clean": (clean_goal, native)}
        for offset in goal_offsets:
            label = f"goal_swap_offset_{offset}"
            conditions[label] = (goal_texts[offset], native)
            conditions[f"{label}_bus_cut"] = (goal_texts[offset], bus_cut)
            conditions[f"{label}_control_cut"] = (goal_texts[offset], control_cut)
        scores = {}
        worlds = {}
        for name, (goal, mask) in conditions.items():
            scores[name], worlds[name] = rank_candidates(model, obs, goal, clean_goal, input_actions, mask, args.clip_reconstruction)
        selected = {name: int(np.argmax(value)) for name, value in scores.items()}
        task_rows = {"suite": args.suite, "task_id": task_id, "task": clean_goal, "goal_texts": goal_texts, "candidate_count": len(candidates), "candidate_seed": args.seed, "candidate_source": args.candidate_source, "candidate_data_root": args.candidate_data_root if args.candidate_source == "recorded" else None, "candidate_task_index": candidate_task_index if args.candidate_source == "recorded" else None, "candidate_episode": candidate_episode if args.candidate_source == "recorded" else None, "scores": {k: v.tolist() for k, v in scores.items()}, "selected": selected}
        task_rows["ranking_flip_rate"] = {name: flip_rate(scores["clean"], value) for name, value in scores.items() if name != "clean"}
        task_rows["replays"] = {}
        candidate_replays = None
        if args.evaluate_all_candidates:
            candidate_replays = []
            for candidate in candidates:
                candidate_replays.append(replay_repeated(task, args.suite, args.seed,
                                                         candidate, args.replay_repeats))
            task_rows["candidate_true_eef_approach_progress"] = [
                float(x["eef_approach_progress"]) for x in candidate_replays
            ]
            task_rows["candidate_true_task_progress_delta"] = [
                float(x["true_progress_delta"]) for x in candidate_replays
            ]
            task_rows["candidate_utility_best_index"] = int(np.argmax(task_rows["candidate_true_eef_approach_progress"]))
        for name, index in selected.items():
            if candidate_replays is not None:
                replay_result = copy.deepcopy(candidate_replays[index])
            else:
                replay_result = replay_repeated(task, args.suite, args.seed,
                                                candidates[index], args.replay_repeats)
            obs_after = replay_result.pop("observations_after_steps")
            ordered_obs = [obs_after[k] for k in (3,) if k in obs_after]
            replay_result["forecast_error_mse"] = forecast_mse(worlds[name][index], ordered_obs, model) if ordered_obs else None
            replay_result["forecast_error_observation_steps"] = [k for k in (3,) if k in obs_after]
            if candidate_replays is not None:
                replay_result["eef_approach_regret_vs_candidate_best"] = float(
                    task_rows["candidate_true_eef_approach_progress"][task_rows["candidate_utility_best_index"]]
                    - replay_result["eef_approach_progress"]
                )
            task_rows["replays"][name] = replay_result
        env.close()
        task_rows["decision_recovery"] = {
            name: selected[name] == selected["clean"]
            for name in selected if name.endswith("_bus_cut") or name.endswith("_control_cut")
        }
        all_rows.append(task_rows)
        print(json.dumps({"task_id": task_id, "selected": selected, "flip_rate": task_rows["ranking_flip_rate"], "replays": task_rows["replays"]}, ensure_ascii=False), flush=True)
    result = {
        "status": "completed_worldagen_counterfactual_action_ranking",
        "forecast_error_protocol": "v2_primary_view_first_chunk_step3_only",
        "planner": "independent_forecast_driven_clip_text_score",
        "native_worldagen_policy_uses_forecast": False,
        "candidate_set_frozen_across_conditions": True,
        "candidate_source": args.candidate_source,
        "selection_target_fixed_to_original_goal": True,
        "selection_score_text": "clean task goal for every condition",
        "clip_reconstruction": args.clip_reconstruction,
        "future_observation_used_for_selection": False,
        "actual_utility_source": "fresh LIBERO replay cumulative sparse reward, success, and environment steps",
        "replay_repeats": args.replay_repeats,
        "replay_repeat_protocol": "independent fresh environment per repeat; top-level metrics are means when repeats > 1",
        "evaluate_all_candidates": args.evaluate_all_candidates,
        "conditions": list(conditions.keys()),
        "goal_offsets": goal_offsets,
        "rows": all_rows,
        "scope": "Minimal independent planner test; no claim about native WorldAgen action selection.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "tasks": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
