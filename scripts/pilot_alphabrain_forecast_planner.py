#!/usr/bin/env python3
"""Small AlphaBrain forecast-driven planner feasibility gate.

The released AlphaBrain policy does not consume its future-world slots for
native action selection. This script therefore evaluates a separate planner:
future-primary forecasts are decoded, scored by frozen CLIP text similarity,
and selected candidates are replayed in the same LIBERO initial state. All
candidate scores are committed before any replay utility is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "LIBERO"))
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))

import pilot_worldagen_counterfactual_action_ranking as worldagen_rank  # noqa: E402


MODEL = ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors"
TEXT = ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_t5_embeddings.pkl"
PRETRAINED = ROOT / "models/AlphaBrain/Cosmos-Predict2-2B-Video2World"
CLIP_WEIGHTS = ROOT / "models/WorldAgen/ViT-B-32.pt"


def _state_from_obs(obs: dict) -> np.ndarray:
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper.size == 1:
        gripper = np.repeat(gripper, 2)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    return np.concatenate([
        gripper[:2], np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3], quat[:4]
    ])


def _image(obs: dict, key: str) -> np.ndarray:
    return np.asarray(
        Image.fromarray(obs[key]).convert("RGB").transpose(Image.Transpose.ROTATE_180).resize((224, 224)),
        dtype=np.uint8,
    )


def _build_input(obs: dict, candidates: torch.Tensor, vae, replace_prop, replace_action):
    primary = torch.from_numpy(_image(obs, "agentview_image")).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    wrist = torch.from_numpy(_image(obs, "robot0_eye_in_hand_image")).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    blank = torch.zeros_like(primary)
    video = torch.cat([blank[:, :1], blank, wrist, primary, blank, blank, wrist, primary, blank], dim=1).unsqueeze(0).cuda()
    with torch.inference_mode():
        latent = vae.encode(video.to(torch.bfloat16) / 127.5 - 1.0).to(torch.bfloat16)
    state = torch.tensor(_state_from_obs(obs)[None], device="cuda", dtype=torch.bfloat16)
    base = replace_prop(latent.clone(), state, torch.tensor([1], device="cuda"))
    batch = base.repeat(candidates.shape[0], 1, 1, 1, 1)
    action = candidates.to(device="cuda", dtype=torch.bfloat16)
    indices = torch.full((candidates.shape[0],), 4, device="cuda", dtype=torch.long)
    batch = replace_action(batch, action, indices)
    cond = torch.zeros((candidates.shape[0], 1, 9, 28, 28), device="cuda", dtype=torch.bfloat16)
    cond[:, :, :5] = 1
    gt = batch[:, :, :5].float().clone()
    return batch, cond, gt


def _load_alpha():
    import safetensors.torch
    sys.path.insert(0, str(ROOT / "vendor" / "AlphaBrain"))
    import pilot_alphabrain_cosmos_policy_goal_swap as fixed  # noqa: E402
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        fixed.install_transformer_engine_stub()
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from AlphaBrain.model.modules.world_model.cosmos.latent_utils import replace_latent_with_action_chunk, replace_latent_with_proprio
    from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper
    model = MinimalV1LVGDiT(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
        out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=2048, num_blocks=28, num_heads=16, crossattn_emb_channels=1024,
        use_crossattn_projection=False, atten_backend="torch", pos_emb_cls="rope3d",
        mlp_ratio=4.0, use_adaln_lora=True, adaln_lora_dim=256, use_wan_fp32_strategy=False,
        pos_emb_learnable=True, rope_enable_fps_modulation=False,
        rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0).cuda().to(dtype=torch.bfloat16).eval()
    sd = safetensors.torch.load_file(str(MODEL), device="cpu")
    sd = {k.removeprefix("dit."): v for k, v in sd.items() if k.startswith("dit.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing learned keys: {len(missing)}")
    del sd
    vae = WanVAEWrapper(str(PRETRAINED), dtype=torch.bfloat16, device="cuda", temporal_window=16).eval().cuda()
    return model, vae, CosmosPolicySampler(), arch_invariant_rand, replace_latent_with_action_chunk, replace_latent_with_proprio, fixed


def _sample_batches(model, vae, sampler, rand, replace_action, replace_prop, obs, candidates, context, cut: bool, fixed):
    # The caller installs the cut once per Goal branch. Re-wrapping the
    # attention modules for every text condition would make a multi-task run
    # depend on wrapper depth rather than on the intervention.
    del cut, fixed
    text = context.to(device="cuda", dtype=torch.bfloat16)
    outputs = []
    batch_size = 2
    for start in range(0, len(candidates), batch_size):
        acts = candidates[start:start + batch_size]
        latent, cond, gt = _build_input(obs, acts, vae, replace_prop, replace_action)
        b = acts.shape[0]
        ctx = text.unsqueeze(0) if text.ndim == 2 else text
        if ctx.shape[0] == 1:
            ctx = ctx.expand(b, -1, -1)
        noise_one = rand((1, 16, 9, 28, 28), seed=1).to(device="cuda", dtype=torch.float32) * 80.0
        noise = noise_one.repeat(b, 1, 1, 1, 1)
        fps = torch.full((b,), 16, device="cuda", dtype=torch.bfloat16)
        padding = torch.zeros((b, 1, 224, 224), device="cuda", dtype=torch.bfloat16)
        def x0_fn(xt, sigma):
            sigma = sigma.float(); t = (sigma.view(-1, 1, 1, 1, 1) / (sigma.view(-1, 1, 1, 1, 1) + 1.0))
            inp = xt.float() * (1.0 - t); inp[:, :, :5] = gt
            frame_noise = (sigma / (sigma + 1.0)).view(-1, 1).expand(-1, 9)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inp.to(torch.bfloat16), frame_noise.to(torch.bfloat16), ctx, cond, fps=fps, padding_mask=padding)
            x0 = (1.0 - t) * xt.float() - t * out.float(); x0[:, :, :5] = gt
            return x0
        with torch.inference_mode():
            outputs.append(sampler.forward(x0_fn=x0_fn, x_sigma_max=noise, num_steps=2, sigma_min=4.0, sigma_max=80.0).detach())
    return torch.cat(outputs, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated task ids; supersedes --task-id")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--init-index", type=int, default=0,
                    help="explicit LIBERO initial-state index; seed alone does not change set_init_state")
    ap.add_argument("--goal-offset", type=int, default=5)
    ap.add_argument("--candidate-data-root", type=Path, default=ROOT / "data/worldagen_goal_probe_extra")
    ap.add_argument("--candidate-map", default="0:5:8,2:3:6,4:0:0,6:1:1,7:4:7,9:2:2")
    ap.add_argument("--output", type=Path, default=ROOT / "runs/alphabrain_forecast_planner_v0/summary.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    from libero.libero import benchmark
    model, vae, sampler, rand, replace_action, replace_prop, fixed = _load_alpha()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if args.task_ids is not None:
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    elif args.task_id is not None:
        task_ids = [int(args.task_id)]
    else:
        task_ids = [0]
    maps = {int(t): (int(d), int(e)) for t, d, e in
            (x.split(":") for x in args.candidate_map.split(","))}
    text_table = pickle.load(open(TEXT, "rb"))
    clip_model, clip_preprocess = clip.load(str(CLIP_WEIGHTS), device="cuda", jit=False)
    clip_model.eval()
    native_cross_attn = [block.cross_attn for block in model.blocks]

    def restore_native_cross_attn():
        for block, original in zip(model.blocks, native_cross_attn):
            block.cross_attn = original

    task_results = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        data_task, episode = maps[task_id]
        env, obs = worldagen_rank.make_env(task, "libero_10", args.seed, args.init_index)
        try:
            candidates = worldagen_rank.recorded_candidates(args.candidate_data_root, data_task, episode, 8, 30)
            score_goal = task.language
            swap_goal = suite.get_task((task_id + args.goal_offset) % suite.get_num_tasks()).language
            contexts = {"clean": text_table[score_goal], "swap": text_table[swap_goal]}
            worlds = {}
            # Native branch and repaired branch use the same sampled noise and
            # candidate batch. Restore original modules before each branch so
            # a multi-task run cannot accumulate intervention wrappers.
            restore_native_cross_attn()
            for label, context in contexts.items():
                worlds[(label, "native")] = _sample_batches(
                    model, vae, sampler, rand, replace_action, replace_prop,
                    obs, candidates[:, :16], context, False, fixed).cpu()
            restore_native_cross_attn()
            fixed.install_fixed_action_world_cut(model)
            for label, context in contexts.items():
                worlds[(label, "cut")] = _sample_batches(
                    model, vae, sampler, rand, replace_action, replace_prop,
                    obs, candidates[:, :16], context, True, fixed).cpu()

            text_feature = clip_model.encode_text(clip.tokenize([score_goal]).cuda()).float()
            text_feature /= text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scores = {}
            for key, latent_cpu in worlds.items():
                with torch.inference_mode():
                    latent = latent_cpu.to(device="cuda", dtype=torch.bfloat16)
                    for name in ("latents_mean", "latents_std", "_inv_std"):
                        if hasattr(vae, name):
                            setattr(vae, name, getattr(vae, name).to(latent.device))
                    decoded = vae.decode(latent).float().clamp(-1, 1)
                    # Slot 7 corresponds to pixel frames 29--32 in the 33-frame layout.
                    images = [Image.fromarray(((decoded[i, :, 29].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)) for i in range(decoded.shape[0])]
                    batch = torch.stack([clip_preprocess(im) for im in images]).cuda()
                    f = clip_model.encode_image(batch).float(); f /= f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    scores[f"{key[0]}_{key[1]}"] = (f @ text_feature.T)[:, 0].cpu().tolist()
            # Commit every score before opening replay environments.
            selected = {k: int(np.argmax(v)) for k, v in scores.items()}
            utilities = []
            for candidate in candidates:
                replay_env, replay_obs = worldagen_rank.make_env(task, "libero_10", args.seed, args.init_index)
                try:
                    utilities.append(worldagen_rank.replay_once(replay_env, replay_obs, candidate)["eef_approach_progress"])
                finally:
                    replay_env.close()
            utilities = [float(x) for x in utilities]
            best = int(np.argmax(utilities))
            task_results.append({
                "task_id": task_id, "seed": args.seed, "init_index": args.init_index, "goal_offset": args.goal_offset,
                "candidate_count": len(candidates), "candidate_horizon_scored": 16,
                "candidate_horizon_replayed": 30, "score_goal": score_goal, "swap_goal": swap_goal,
                "scores": scores, "selected": selected, "utilities_eef_progress": utilities,
                "regret": {k: float(utilities[best] - utilities[i]) for k, i in selected.items()},
                "best_candidate": best,
            })
            print(json.dumps({"task_id": task_id, "selected": selected,
                              "best_candidate": best,
                              "regret": task_results[-1]["regret"]}), flush=True)
        finally:
            env.close()
            restore_native_cross_attn()
            torch.cuda.empty_cache()
    if len(task_results) == 1:
        result = {
            "status": "completed_alphabrain_forecast_planner_feasibility",
            **task_results[0],
            "candidate_data_root": str(args.candidate_data_root),
            "candidate_map": args.candidate_map,
            "native_policy_uses_forecast": False,
            "scope_limits": ["Independent forecast-driven planner only; one task-seed feasibility gate.", "CLIP scores are selection-only; replay EEF is the utility.", "No native-policy or task-success claim."],
        }
    else:
        result = {
            "status": "completed_alphabrain_forecast_planner_multitask_feasibility",
            "task_ids": task_ids, "seed": args.seed, "init_index": args.init_index, "goal_offset": args.goal_offset,
            "candidate_data_root": str(args.candidate_data_root), "candidate_map": args.candidate_map,
            "task_results": task_results, "native_policy_uses_forecast": False,
            "aggregate": {
                "selected_action_change_rate": float(np.mean([r["selected"]["clean_native"] != r["selected"]["swap_native"] for r in task_results])),
                "clean_regret_mean_m": float(np.mean([r["regret"]["clean_native"] for r in task_results])),
                "swap_regret_mean_m": float(np.mean([r["regret"]["swap_native"] for r in task_results])),
                "clean_cut_regret_mean_m": float(np.mean([r["regret"]["clean_cut"] for r in task_results])),
                "swap_cut_regret_mean_m": float(np.mean([r["regret"]["swap_cut"] for r in task_results])),
                "bus_cut_clean_recovery_rate": float(np.mean([r["selected"]["clean_cut"] == r["selected"]["clean_native"] for r in task_results])),
                "bus_cut_swap_score_identity": all(r["scores"]["clean_cut"] == r["scores"]["swap_cut"] for r in task_results),
            },
            "scope_limits": ["Independent forecast-driven planner only; multi-task feasibility gate.", "CLIP scores are selection-only; replay EEF is the utility.", "No native-policy or task-success claim."],
        }
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if "aggregate" in result:
        print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))
    else:
        print(json.dumps({k: result[k] for k in ("status", "selected", "best_candidate", "regret")}, indent=2))


if __name__ == "__main__":
    main()
