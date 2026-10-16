#!/usr/bin/env python3
"""Test activation-patch repair in the independent WorldAgen action planner.

The native controller does not consume WorldAgen forecasts.  This script uses
the same frozen recorded-action candidate set as the counterfactual ranking
test, but compares a swapped-goal forward whose intermediate action-token
states are replaced with the corresponding Clean-goal states.  The patch is
an internal repair diagnostic, not a native-controller attack claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import clip
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "LIBERO"))
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_worldagen_counterfactual_action_ranking as ranking  # noqa: E402
import pilot_worldagen_goal_bus_replicate as probe  # noqa: E402


def run_with_action_capture(model, batch, mask, capture_layer: int,
                            patch_source: torch.Tensor | None = None):
    """Run one batch, optionally replacing action-token states at one layer."""
    model.attention_mask.copy_(mask)
    action_positions = []
    block = int(mask.shape[0] // model.sequence_length)
    for chunk in range(model.sequence_length):
        start = chunk * block
        action_positions.extend(range(
            start + 1 + model.state_sequence_length,
            start + 1 + model.state_sequence_length + model.action_sequence_length,
        ))
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0]
        captured["action"] = hidden[:, action_positions, :].detach().clone()
        if patch_source is not None:
            patched = hidden.clone()
            patched[:, action_positions, :] = patch_source
            return (patched,) + tuple(output[1:])
        return output

    handle = model.transformer_backbone.layers[capture_layer].register_forward_hook(hook)
    torch.manual_seed(20261011)
    if probe.DEVICE == "cuda":
        torch.cuda.manual_seed_all(20261011)
    primary, wrist, state, text, actions = batch
    with torch.inference_mode():
        arm, grip, world, _, _ = model(primary, wrist, state, text, actions)
    handle.remove()
    policy = torch.cat((arm, grip), dim=-1).detach().clone()
    return world.detach().clone(), policy, captured["action"]


def world_primary(world: torch.Tensor, n: int) -> torch.Tensor:
    return ranking.primary_forecasts(world, n).detach().cpu()


@torch.inference_mode()
def score_worlds(model, obs: dict, worlds: torch.Tensor, goal_text: str,
                 reconstruction: str) -> np.ndarray:
    text = clip.tokenize([goal_text]).to(probe.DEVICE)
    text_feature = model.clip_model.encode_text(text).float()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference = None
    if reconstruction == "reference_patch":
        from PIL import Image
        reference = model.image_processor(
            Image.fromarray(obs["agentview_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180)
        ).to(probe.DEVICE)
    scores = []
    for world in worlds:
        image = model.image_processor(
            ranking.unpatch_for_clip(world[0], reference)
        ).unsqueeze(0).to(probe.DEVICE)
        image_feature = model.clip_model.encode_image(image).float()
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores.append(float((image_feature * text_feature).sum().item()))
    return np.asarray(scores, dtype=np.float64)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-ids", default="0,2")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--candidate-data-root", required=True)
    ap.add_argument("--candidate-map", required=True)
    ap.add_argument("--recorded-replay-horizon", type=int, default=30)
    ap.add_argument("--patch-layers", default="4,8")
    ap.add_argument("--clip-reconstruction", choices=["global_minmax", "reference_patch"], default="global_minmax")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the WorldAgen activation-patch ranking test")
    torch.set_num_threads(4)
    model = probe.build_model()
    native, bus_cut = ranking.libero.masks(model)
    candidate_map = {}
    for item in args.candidate_map.split(","):
        tid, did, episode = item.split(":")
        candidate_map[int(tid)] = (int(did), int(episode))
    patch_layers = [int(x) for x in args.patch_layers.split(",") if x.strip()]
    suite = ranking.libero.benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.task_ids.split(",") if x]
    rows = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env, obs = ranking.make_env(task, args.suite, args.seed)
        clean_goal = task.language
        num_tasks = suite.get_num_tasks()
        swapped_goal = suite.get_task((task_id + 1) % num_tasks).language
        data_task, episode = candidate_map[task_id]
        candidates = ranking.recorded_candidates(
            Path(args.candidate_data_root), data_task, episode,
            args.candidates, args.recorded_replay_horizon,
        )
        input_actions = candidates[:, :args.horizon]
        if input_actions.shape[1] < 6:
            input_actions = torch.cat(
                [input_actions, input_actions[:, -1:].repeat(1, 6 - input_actions.shape[1], 1)], dim=1
            )
        clean_batch = ranking.batch_from_observation(model, obs, clean_goal, input_actions)
        swap_batch = ranking.batch_from_observation(model, obs, swapped_goal, input_actions)
        clean_world_raw, _, clean_action_state = run_with_action_capture(
            model, clean_batch, native, patch_layers[0]
        )
        swap_world_raw, _, _ = run_with_action_capture(
            model, swap_batch, native, patch_layers[0]
        )
        conditions = {
            "clean": world_primary(clean_world_raw, len(candidates)),
            "goal_swap": world_primary(swap_world_raw, len(candidates)),
            "goal_swap_bus_cut": world_primary(
                run_with_action_capture(model, swap_batch, bus_cut, patch_layers[0])[0], len(candidates)
            ),
        }
        for layer in patch_layers:
            # Capture a clean source at this exact layer before patching the
            # swapped-goal forward, so each repair is layer-matched.
            clean_raw, _, source = run_with_action_capture(model, clean_batch, native, layer)
            patched_raw, _, _ = run_with_action_capture(
                model, swap_batch, native, layer, patch_source=source
            )
            conditions[f"goal_swap_activation_patch_l{layer}"] = world_primary(
                patched_raw, len(candidates)
            )
            # Keep the clean layer output available for diagnostics without
            # changing the planner's candidate set.
            del clean_raw
        scores = {
            name: score_worlds(model, obs, worlds, clean_goal, args.clip_reconstruction)
            for name, worlds in conditions.items()
        }
        clean_reference = conditions["clean"].double()
        forecast_residual = {
            name: [
                float((worlds[i].double() - clean_reference[i]).norm()
                      / clean_reference[i].norm().clamp_min(1e-12))
                for i in range(len(candidates))
            ]
            for name, worlds in conditions.items()
        }
        selected = {name: int(np.argmax(values)) for name, values in scores.items()}
        task_row = {
            "suite": args.suite,
            "task_id": task_id,
            "task": clean_goal,
            "swapped_goal": swapped_goal,
            "candidate_count": len(candidates),
            "candidate_seed": args.seed,
            "candidate_source": "recorded",
            "candidate_task_index": data_task,
            "candidate_episode": episode,
            "clip_reconstruction": args.clip_reconstruction,
            "scores": {k: v.tolist() for k, v in scores.items()},
            "forecast_relative_l2_to_clean": forecast_residual,
            "selected": selected,
            "ranking_flip_rate": {
                name: flip_rate(scores["clean"], values)
                for name, values in scores.items() if name != "clean"
            },
            "replays": {},
        }
        # Replay every selected trajectory from the same fresh initial state;
        # no predicted or future observation enters selection.
        for name, index in selected.items():
            replay = ranking.replay_repeated(task, args.suite, args.seed,
                                             candidates[index], repeats=1)
            obs_after = replay.pop("observations_after_steps")
            ordered = [obs_after[k] for k in (3,) if k in obs_after]
            replay["forecast_error_mse"] = ranking.forecast_mse(
                conditions[name][index], ordered, model
            ) if ordered else None
            replay["forecast_error_observation_steps"] = [k for k in (3,) if k in obs_after]
            task_row["replays"][name] = replay
        task_row["decision_recovery"] = {
            name: selected[name] == selected["clean"]
            for name in selected
            if name.startswith("goal_swap_activation_patch") or name.endswith("_bus_cut")
        }
        rows.append(task_row)
        env.close()
        print(json.dumps({"task_id": task_id, "selected": selected,
                          "flip_rate": task_row["ranking_flip_rate"]}, ensure_ascii=False), flush=True)
    result = {
        "status": "completed_worldagen_activation_patch_action_ranking",
        "planner": "independent_forecast_driven_clip_text_score",
        "native_worldagen_policy_uses_forecast": False,
        "candidate_set_frozen_across_conditions": True,
        "selection_target_fixed_to_original_goal": True,
        "future_observation_used_for_selection": False,
        "activation_patch_layers": patch_layers,
        "activation_patch_source": "same-candidate Clean-goal action-token hidden states",
        "forecast_error_protocol": "v2_primary_view_first_chunk_step3_only",
        "clip_reconstruction": args.clip_reconstruction,
        "actual_utility_source": "fresh LIBERO replay simulator state and sparse reward",
        "conditions": list(rows[0]["selected"].keys()) if rows else [],
        "rows": rows,
        "scope": "Activation-patch repair diagnostic; no claim about native WorldAgen action selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "tasks": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
