#!/usr/bin/env python3
"""Official NVIDIA Cosmos Policy checkpoint Clean full-episode LIBERO gate.

N=1 uses joint native action generation. N>1 uses the released depth-one
best-of-N: autoregressive fixed-action future, autoregressive future-conditioned
value, select maximum value. All calls retain the original task goal. This is
a torch-backend adapter, with explicit provenance; success is environment-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import cosmos_policy_native_inference as inference

ROOT = Path(__file__).resolve().parents[1]
LIBERO = ROOT / "vendor/LIBERO"
CKPT_DIR = inference.MODEL.parent
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(LIBERO))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_ids(value):
    ids = [int(item) for item in value.split(",")]
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("expected unique nonnegative comma-separated IDs")
    return ids


def frame(obs):
    # Display the same camera orientation passed into Cosmos preprocessing.
    return np.concatenate([
        np.ascontiguousarray(obs[key][::-1])
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    ], axis=1)


def run_episode(model, vae, sampler, rand, task, task_id, init_id, text, stats,
                args, provenance):
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    bddl = LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    states = torch.load(init_path, map_location="cpu", weights_only=False)
    initial = np.asarray(states[init_id])
    label = f"{args.suite}_task{task_id:02d}_init{init_id:02d}_clean"
    output = args.out_dir
    trace_path, chunks_path = output / f"{label}.jsonl", output / f"{label}_chunks.jsonl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0)
    env.task_id, env.task_name, env.task_suite_name = task_id, task.name, args.suite
    result = {
        "status": "running", "model": "NVIDIA Cosmos Policy LIBERO 40K (torch adapter)",
        "condition": "clean", "suite": args.suite, "best_of_n": args.best_of_n,
        "task_id": task_id, "init_id": init_id, "task_name": task.name,
        "goal": task.language, "max_steps": args.max_steps, "settle_steps": args.settle_steps,
        "seed": args.seed, "environment_seed": args.env_seed,
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "bddl": str(bddl), "bddl_sha256": digest(bddl), "init_file_sha256": digest(init_path),
        "trace": str(trace_path), "chunks": str(chunks_path), "provenance": provenance,
    }
    path = output / f"{label}.json"
    atomic_json(path, result)
    writer = None
    steps, chunks, success = 0, 0, False
    started = time.monotonic()
    rewards, action_norms, eef_path = [], [], []
    torch.cuda.reset_peak_memory_stats()
    try:
        env.seed(args.env_seed)
        env.reset()
        obs = env.set_init_state(states[init_id])
        initial_sim = np.asarray(env.get_sim_state()).copy()
        dummy = [0.0] * 6 + [-1.0]
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(dummy)
        settled_sim = np.asarray(env.get_sim_state()).copy()
        result["success_at_start"] = bool(env.check_success())
        if result["success_at_start"]:
            raise RuntimeError("Task already successful before first policy action; invalid clean episode")
        Image.fromarray(frame(obs)).save(output / f"{label}_initial.png")
        if args.video_stride:
            writer = imageio.get_writer(str(output / f"{label}.mp4"), fps=20 / args.video_stride,
                                        codec="libx264", quality=7, macro_block_size=16)
            writer.append_data(frame(obs))
        embedding = text[task.language].to(device="cuda", dtype=torch.bfloat16)
        with trace_path.open("w") as trace, chunks_path.open("w") as chunk_log:
            done = False
            while steps < args.max_steps:
                query_started = time.monotonic()
                candidate_log = None
                if args.best_of_n == 1:
                    normalized, latent = inference.query(
                        model, vae, sampler, rand, obs, embedding, stats, num_steps=5,
                        return_latent=True, image_transform="vertical", seed=args.seed,
                    )
                    actions = inference.normalize_action(normalized, stats)[0]
                else:
                    selection = inference.best_of_n(
                        model, vae, sampler, rand, obs, embedding, stats,
                        n=args.best_of_n, seed=args.seed,
                    )
                    selected = selection["candidates"][selection["selected_index"]]
                    normalized, latent = selected["normalized_actions"], selected["future_latent"]
                    actions = selected["actions"][0]
                    candidate_log = {
                        "selected_index": selection["selected_index"],
                        "candidates": [{"index": item["index"], "seed": item["seed"],
                                        "predicted_value": item["value"],
                                        "environment_actions": item["actions"][0].tolist()}
                                       for item in selection["candidates"]],
                    }
                    del selection, selected
                if actions.shape != (16, 7) or not np.isfinite(actions).all() or not torch.isfinite(latent).all():
                    raise RuntimeError("Nonfinite or malformed policy output")
                chunks += 1
                chunk_log.write(json.dumps({
                    "chunk": chunks, "first_step": steps + 1,
                    "normalized_actions": normalized.float().cpu().numpy()[0].tolist(),
                    "environment_actions": actions.tolist(),
                    "query_seconds": time.monotonic() - query_started,
                    "future_slot_rms": latent[:, :, 5:8].float().square().mean().sqrt().item(),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "native_best_of_n": candidate_log,
                }, allow_nan=False) + "\n")
                chunk_log.flush()
                del latent, normalized
                for action in actions[:args.max_steps - steps]:
                    before = np.asarray(obs["robot0_eef_pos"]).copy()
                    obs, reward, done, _ = env.step(action.tolist())
                    steps += 1
                    success = bool(env.check_success())
                    after = np.asarray(obs["robot0_eef_pos"]).copy()
                    rewards.append(float(reward))
                    action_norms.append(float(np.linalg.norm(action)))
                    eef_path.append(after)
                    trace.write(json.dumps({
                        "step": steps, "chunk": chunks, "action": action.tolist(),
                        "eef_before": before.tolist(), "eef_after": after.tolist(),
                        "eef_quaternion": np.asarray(obs["robot0_eef_quat"]).tolist(),
                        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                        "reward": float(reward), "done": bool(done), "task_success": success,
                    }, allow_nan=False) + "\n")
                    if writer is not None and (steps % args.video_stride == 0 or success or done):
                        writer.append_data(frame(obs))
                    if success or done:
                        break
                trace.flush()
                print(json.dumps({"episode": label, "steps": steps, "chunks": chunks,
                                  "success": success, "elapsed_seconds": time.monotonic() - started}), flush=True)
                if success or done:
                    break
        Image.fromarray(frame(obs)).save(output / f"{label}_final.png")
        np.savez_compressed(output / f"{label}_states.npz", initial=initial_sim, settled=settled_sim,
                            final=np.asarray(env.get_sim_state()), eef_path=np.asarray(eef_path))
        result.update(status="completed", success=success, terminal_done=bool(done),
                      environment_steps=steps, action_chunks=chunks, reward_sum=sum(rewards),
                      final_reward=rewards[-1] if rewards else None,
                      action_norm_mean=float(np.mean(action_norms)) if action_norms else None,
                      initial_image=str(output / f"{label}_initial.png"),
                      final_image=str(output / f"{label}_final.png"),
                      video=str(output / f"{label}.mp4") if writer is not None else None)
        return result
    except BaseException as error:
        result.update(status="error", exception_type=type(error).__name__, exception=str(error),
                      environment_steps=steps, action_chunks=chunks)
        raise
    finally:
        if writer is not None:
            writer.close()
        result.update(elapsed_seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(path, result)
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", type=parse_ids, default=[0])
    parser.add_argument("--init-ids", type=parse_ids, default=[0])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--best-of-n", type=int, default=1)
    parser.add_argument("--offload", action="store_true",
                        help="Keep weights on CPU; compute each DiT block and VAE on CUDA sequentially")
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--video-stride", type=int, default=2, help="0 disables video; PNGs always saved")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                        help="Optional allocator cap as fraction of device VRAM; 0 uses PyTorch default")
    parser.add_argument("--preflight", action="store_true", help="CPU-only resource/protocol checks; no environment or GPU")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.best_of_n < 1 or args.max_steps < 1 or args.settle_steps < 0 or args.video_stride < 0 or not 0 <= args.cuda_memory_fraction <= 1:
        parser.error("invalid horizon, settling, video stride, or allocator fraction")
    args.out_dir = args.out_dir.resolve()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    text_path, stats_path = CKPT_DIR / "libero_t5_embeddings.pkl", CKPT_DIR / "libero_dataset_statistics.json"
    with text_path.open("rb") as handle:
        text = pickle.load(handle)
    inference.verify_artifact(text_path, inference.TEXT_SHA256)
    stats = inference.load_stats(stats_path)
    episodes = []
    for task_id in args.task_ids:
        if task_id >= suite.n_tasks:
            parser.error(f"task {task_id} outside suite of {suite.n_tasks}")
        task = suite.get_task(task_id)
        if task.language not in text or not torch.isfinite(text[task.language]).all():
            raise ValueError(f"Missing/nonfinite task text embedding: {task.language}")
        states = suite.get_task_init_states(task_id)
        for init_id in args.init_ids:
            if init_id >= len(states):
                parser.error(f"init {init_id} outside task's {len(states)} states")
            episodes.append({"task_id": task_id, "init_id": init_id, "goal": task.language})
    inference.verify_artifact(inference.MODEL, inference.MODEL_SHA256)
    inference.verify_artifact(inference.TOKENIZER, inference.TOKENIZER_SHA256)
    provenance = {
        "checkpoint": str(inference.MODEL), "checkpoint_sha256": inference.MODEL_SHA256,
        "checkpoint_repo": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        "checkpoint_revision": inference.CHECKPOINT_REVISION,
        "checkpoint_training_scope": "LIBERO all four suites, 40K steps; official model card declares 50/25/25 policy/world/value",
        "upstream_source_revision": inference.SOURCE_REVISION,
        "tokenizer": str(inference.TOKENIZER), "tokenizer_sha256": inference.TOKENIZER_SHA256,
        "tokenizer_source": "Public original Wan-AI tokenizer is byte-identical to NVIDIA-required tokenizer",
        "vae": "official tokenizer tensors, source normalization constants bf16, inherited temporal_window=16, is_amp=False",
        "helper_sha256": digest(Path(inference.__file__)), "runner_sha256": digest(Path(__file__)),
        "source_sha256": {
            relative: digest(ROOT / relative) for relative in (
                "scripts/cosmos_policy_codecs.py",
                "scripts/cosmos_policy_wan_vae.py",
                "scripts/cosmos_policy_torch_compat.py",
                "scripts/alphabrain_checkpoint_inference.py",
                "scripts/pilot_alphabrain_cosmos_policy_goal_swap.py",
                "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/cosmos_sampler.py",
                "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/official_dit/minimal_v4_dit.py",
            )
        },
        "stats_sha256": digest(stats_path), "text_embeddings_sha256": digest(text_path),
        "image_codec": "official raw256 -> vertical flip -> JPEG95 -> resize224 -> 90% center-crop -> resize224",
        "action_codec": "official inverse min-max with no clipping; continuous gripper; no sign inversion",
        "proprio_codec": "official minmax concatenated observation gripper_qpos,eef_pos,eef_quat; float64 stats",
        "settle_action": [0.0] * 6 + [-1.0], "chunk_length": 16,
        "action_steps": 5, "future_steps": 1 if args.best_of_n > 1 else None,
        "value_steps": 1 if args.best_of_n > 1 else None,
        "sampler_seeds": [args.seed + i for i in range(args.best_of_n)],
        "sigma_min": 4.0, "sigma_max": 80.0, "condition_timestep": 0.0,
        "policy_condition_slots": [0, 1, 2, 3],
        "native_ar_forecast_condition_slots": [0, 1, 2, 3, 4] if args.best_of_n > 1 else None,
        "native_ar_value_condition_slots": list(range(8)) if args.best_of_n > 1 else None,
        "selection": "native argmax predicted value" if args.best_of_n > 1 else "single policy proposal",
        "goal_routing": "original task goal in all policy, forecast, and value calls",
        "attention_backend": "torch with existing TE compatibility shim",
        "runtime_limit": "Strict source/weight adapter; full upstream Transformer Engine execution parity not established",
        "cudnn_deterministic": True,
        "success_source": "LIBERO environment check_success() after each physical action",
        "external_planner": False, "training_updates": 0, "api_requests": 0,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    if args.preflight:
        print(json.dumps({"status": "cpu_preflight_passed", "episodes": episodes, "provenance": provenance}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for native full-task inference")
    # Match upstream set_seed_everywhere before any tokenizer/model CUDA work.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        parser.error("output directory is nonempty; choose a fresh directory")
    if args.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    started = time.monotonic()
    summary = {"status": "loading", "model": "NVIDIA Cosmos Policy LIBERO 40K (torch adapter)",
               "condition": "clean", "provenance": provenance,
               "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "planned_episodes": episodes, "results": []}
    path = args.out_dir / "summary.json"
    atomic_json(path, summary)
    try:
        model, vae, sampler, rand = inference.load_model(device="cpu" if args.offload else "cuda")
        if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
            from cosmos_policy_torch_compat import compatibility_metadata
            provenance["runtime_compatibility"] = compatibility_metadata()
        else:
            provenance["runtime_compatibility"] = {"backend": "native Transformer Engine",
                                                    "full_runtime_parity_verified": False}
        if args.offload:
            from cosmos_policy_cuda_offload import prepare
            model, vae, offload_manager, offload_metadata = prepare(model, vae)
            provenance["execution_offload"] = offload_metadata
            provenance["offload_helper_sha256"] = digest(ROOT / "scripts/cosmos_policy_cuda_offload.py")
            print(json.dumps({"offload_preflight": offload_metadata}), flush=True)
        summary["status"] = "running"
        for item in episodes:
            result = run_episode(model, vae, sampler, rand, suite.get_task(item["task_id"]),
                                 item["task_id"], item["init_id"], text, stats, args, provenance)
            summary["results"].append(result)
            successes = sum(row["success"] for row in summary["results"])
            count = len(summary["results"])
            summary["aggregate"] = {"episodes": count, "successes": successes, "success_rate": successes / count}
            summary["elapsed_seconds"] = time.monotonic() - started
            atomic_json(path, summary)
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_json(path, summary)
    print(json.dumps({"summary": str(path), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
