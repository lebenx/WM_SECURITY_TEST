#!/usr/bin/env python3
"""Short, frozen-weight goal-free context baseline for AlphaBrain.

This is a deliberately bounded baseline, not a replacement checkpoint:
the released 2B DiT and VAE stay frozen and only one shared 512x1024
instruction-context tensor is optimized on held-out latent forecast error.
Training and evaluation episodes are disjoint.  The result answers whether
the current exact edge-cut evidence is merely an untrained zero-context
artifact; it does not claim task capability or policy utility.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# The frozen WorldAgen environment uses CUDA torch 2.2, while recent
# diffusers probes for an optional Intel XPU namespace during import.
if not hasattr(torch, "xpu"):
    class _NoXPU:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): return None
        @staticmethod
        def device_count(): return 0
        @staticmethod
        def manual_seed(_seed): return None
        @staticmethod
        def manual_seed_all(_seed): return None
        @staticmethod
        def _is_in_bad_fork(): return False
    torch.xpu = _NoXPU()

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "ALPHABRAIN_GOALFREE_OUT",
    str(ROOT / "runs/alphabrain_goal_free_context_v0/summary.json"),
))
DATA_FILES = [
    Path(x) for x in os.environ.get(
        "ALPHABRAIN_GOALFREE_DATA_FILES",
        ",".join(str(p) for p in sorted(
            (ROOT / "data/worldagen_goal_probe_extra/data/chunk-000").glob("file-*.parquet")
        )),
    ).split(",") if x
]
BASE_FRAMES = [int(x) for x in os.environ.get("ALPHABRAIN_GOALFREE_FRAMES", "30").split(",")]
TRAIN_EPISODES = {int(x) for x in os.environ.get("ALPHABRAIN_GOALFREE_TRAIN_EPISODES", "3,4,5").split(",") if x}
EVAL_EPISODES = {int(x) for x in os.environ.get("ALPHABRAIN_GOALFREE_EVAL_EPISODES", "6,7,8,9").split(",") if x}
STEPS = int(os.environ.get("ALPHABRAIN_GOALFREE_STEPS", "8"))
LR = float(os.environ.get("ALPHABRAIN_GOALFREE_LR", "0.05"))
SEED = int(os.environ.get("ALPHABRAIN_GOALFREE_SEED", "17"))


def load_rows_and_items():
    import pyarrow.parquet as pq

    # Reuse the audited preparation path so image flips, VAE encoding, state
    # ordering, action injection, target selection, and noise are identical to
    # the released forecast matrix.
    sys.path.insert(0, str(ROOT / "scripts"))
    import pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix as matrix

    rows = []
    for path in DATA_FILES:
        rows.extend(pq.read_table(path).to_pylist())
    task_path = DATA_FILES[0].parents[2] / "meta/tasks.parquet"
    task_names = [r.get("task", r.get("__index_level_0__")) for r in pq.read_table(task_path).to_pylist()]
    candidates = []
    for episode in sorted({int(r["episode_index"]) for r in rows}):
        indexed = {int(r["frame_index"]): r for r in rows if int(r["episode_index"]) == episode}
        for frame in BASE_FRAMES:
            if frame in indexed:
                candidates.append(indexed[frame])

    model, vae, _sampler, rand, replace_prop, replace_action, _fixed = matrix.load_arm()
    items = []
    for row in candidates:
        item = matrix.prepare(rows, row, task_names, vae, replace_prop, replace_action, rand)
        items.append(item)
    del vae
    return model, items


def one_step(model, item, context):
    """Differentiable sigma=80 forecast used for both fitting and evaluation."""
    xt = item["noise"].float()
    gt = item["gt"].float()
    sigma = torch.tensor([80.0], device=xt.device, dtype=torch.float32)
    t = sigma.view(1, 1, 1, 1, 1) / 81.0
    net_input = xt * (1.0 - t)
    net_input[:, :, :5] = gt
    frame_noise = (sigma / 81.0).view(1, 1).expand(1, 9)
    cond = item["cond"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        net_output = model(
            net_input.to(torch.bfloat16),
            frame_noise.to(torch.bfloat16),
            context.to(torch.bfloat16),
            cond,
            fps=item["fps"],
            padding_mask=item["padding"],
        )
    forecast = (1.0 - t) * xt - t * net_output.float()
    forecast[:, :, :5] = gt
    return forecast


def mse(model, item, context):
    pred = one_step(model, item, context)
    target = item["target"].float()[:, :, 5:8]
    return (pred[:, :, 5:8] - target).pow(2).mean()


def summarize(model, items, context):
    values = []
    with torch.no_grad():
        for item in items:
            values.append(float(mse(model, item, context).item()))
    return {
        "windows": len(values),
        "episodes": sorted({int(item["episode"]) for item in items}),
        "mse_mean": float(np.mean(values)) if values else None,
        "mse_values": values,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the bounded goal-free baseline")
    started = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, items = load_rows_and_items()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_items = [x for x in items if int(x["episode"]) in TRAIN_EPISODES]
    eval_items = [x for x in items if int(x["episode"]) in EVAL_EPISODES]
    if not train_items or not eval_items:
        raise RuntimeError(f"empty split: train={len(train_items)} eval={len(eval_items)}")

    # The learned context starts at the released zero-context diagnostic and
    # is the only trainable object.  Parameter is fp32 for stable Adam updates;
    # the frozen DiT receives its normal bf16 context input.
    context = nn.Parameter(torch.zeros_like(items[0]["contexts"][2], dtype=torch.float32))
    optimizer = torch.optim.Adam([context], lr=LR)
    initial = summarize(model, eval_items, context.detach())
    losses = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        item = train_items[step % len(train_items)]
        loss = mse(model, item, context)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([context], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        print(json.dumps({"step": step + 1, "train_mse": losses[-1]}), flush=True)

    learned = summarize(model, eval_items, context.detach())
    result = {
        "status": "completed_bounded_goal_free_context_baseline",
        "method": "frozen_AlphaBrain_DiT_plus_one_shared_trainable_null_context",
        "checkpoint": "models/AlphaBrain/cosmos_policy_libero/model.safetensors",
        "train_episodes": sorted(TRAIN_EPISODES),
        "eval_episodes": sorted(EVAL_EPISODES),
        "train_windows": len(train_items),
        "eval_windows": len(eval_items),
        "base_frames": BASE_FRAMES,
        "steps": STEPS,
        "learning_rate": LR,
        "seed": SEED,
        "initial_zero_context_eval": initial,
        "learned_context_eval": learned,
        "train_mse_trace": losses,
        "context_l2": float(context.detach().norm().item()),
        "scope": "A bounded latent one-step baseline with disjoint episode split; no policy fine-tuning, no environment rollout, and no claim that a shared learned context is a full trained goal-free dynamics head.",
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "initial_eval_mse": initial["mse_mean"],
        "learned_eval_mse": learned["mse_mean"],
        "steps": STEPS,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
