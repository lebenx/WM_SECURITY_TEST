#!/usr/bin/env python3
"""AlphaBrain action-carrier activation mediation on fixed real inputs.

The released DiT uses temporal slots 0--8: condition 0:4, action 4, future
world 5:8, and value 8.  For each real LIBERO window we run the native A/B
goal pair with identical latent, recorded action, timestep, and sampler noise.
At selected DiT blocks, a paired A/B denoiser forward replaces only B's action
slot hidden state with A's.  The resulting sampled world residual tests
whether the intermediate action carrier mediates the goal effect.  This is an
internal diagnostic, not an attacker operation or task-level claim.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix as base  # noqa: E402

OUT = Path(os.environ.get(
    "ALPHABRAIN_ACTIVATION_PATCH_OUT",
    str(ROOT / "runs/alphabrain_action_carrier_activation_patch_v0/summary.json"),
))
LAYERS = [int(x) for x in os.environ.get("ALPHABRAIN_ACTIVATION_PATCH_LAYERS", "4,8,12,16,20,24,27").split(",") if x]
PATCH_SLOTS = [int(x) for x in os.environ.get("ALPHABRAIN_ACTIVATION_PATCH_SLOTS", "4,8").split(",") if x]
EPISODES = [int(x) for x in os.environ.get("ALPHABRAIN_ACTIVATION_PATCH_EPISODES", "3,4,5").split(",") if x]


def paired_sample(model, sampler, item, patch_layer: int | None = None):
    """Run A/B together, optionally copying A's action slot into B."""
    latent = torch.cat([item["latent"], item["latent"]], dim=0)
    gt = torch.cat([item["gt"], item["gt"]], dim=0)
    cond = item["cond"].repeat(2, 1, 1, 1, 1)
    padding = item["padding"].repeat(2, 1, 1, 1)
    fps = item["fps"].repeat(2)
    noise = item["noise"].repeat(2, 1, 1, 1, 1)
    contexts = torch.cat([item["contexts"][0], item["contexts"][1]], dim=0)
    handles = []
    if patch_layer is not None:
        def hook(_module, _inputs, output):
            patched = output.clone()
            # x is B,T,H,W,D; slots 4 and 8 are the action/value carriers.
            for slot in PATCH_SLOTS:
                patched[1:2, slot:slot + 1] = output[0:1, slot:slot + 1]
            return patched
        # The carrier can be rewritten by every later cross-attention block.
        # Keep the intervention active from the selected layer onward so the
        # counterfactual tests persistent mediation rather than one-block
        # overwrite-and-decay.
        for idx in range(patch_layer, len(model.blocks)):
            handles.append(model.blocks[idx].register_forward_hook(hook))

    def x0_fn(xt, sigma):
        sigma = sigma.float()
        s = sigma.view(-1, 1, 1, 1, 1)
        t = s / (s + 1.0)
        inp = xt.float() * (1.0 - t)
        inp[:, :, :5] = gt
        frame_noise = (sigma / (sigma + 1.0)).view(-1, 1).expand(-1, 9)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp.to(torch.bfloat16), frame_noise.to(torch.bfloat16), contexts,
                        cond, fps=fps, padding_mask=padding)
        x0 = (1.0 - t) * xt.float() - t * out.float()
        x0[:, :, :5] = gt
        return x0

    try:
        with torch.inference_mode():
            return sampler.forward(x0_fn=x0_fn, x_sigma_max=noise,
                                   num_steps=2, sigma_min=4.0, sigma_max=80.0)
    finally:
        for handle in handles:
            handle.remove()


def slot_delta(a: torch.Tensor, b: torch.Tensor, slots: list[int]) -> float:
    return float((a.float()[:, :, slots] - b.float()[:, :, slots]).abs().mean().item())


def main() -> None:
    started = time.monotonic()
    torch.set_grad_enabled(False)
    import pyarrow.parquet as pq

    all_rows = []
    for path in base.DATA_FILES:
        all_rows.extend(pq.read_table(path).to_pylist())
    task_table = pq.read_table(base.DATA_FILES[0].parents[2] / "meta/tasks.parquet").to_pylist()
    task_names = [row.get("task", row.get("__index_level_0__")) for row in task_table]
    rows_by_ep = {}
    for row in all_rows:
        rows_by_ep.setdefault(int(row["episode_index"]), []).append(row)
    candidates = []
    for episode in EPISODES:
        ep_rows = {int(row["frame_index"]): row for row in rows_by_ep.get(episode, [])}
        if 30 in ep_rows:
            candidates.append(ep_rows[30])
    if not candidates:
        raise RuntimeError(f"no requested episodes among {EPISODES}")

    model, vae, sampler, rand, replace_prop, replace_action, fixed = base.load_arm()
    items = [base.prepare(all_rows, row, task_names, vae, replace_prop, replace_action, rand) for row in candidates]
    # Native paired output is the common denominator for every layer patch.
    native = []
    for item in items:
        native.append(paired_sample(model, sampler, item, None).cpu())
    rows = []
    for item, nat in zip(items, native):
        layer_results = []
        native_world = slot_delta(nat[0:1], nat[1:2], [5, 6, 7])
        native_action = slot_delta(nat[0:1], nat[1:2], [4])
        native_value = slot_delta(nat[0:1], nat[1:2], [8])
        for layer in LAYERS:
            patched = paired_sample(model, sampler, item, layer).cpu()
            patched_world = slot_delta(patched[1:2], nat[0:1], [5, 6, 7])
            patched_action = slot_delta(patched[1:2], nat[0:1], [4])
            patched_value = slot_delta(patched[1:2], nat[0:1], [8])
            layer_results.append({
                "layer": layer,
                "patched_world_to_recorded_abs": patched_world,
                "patched_action_to_recorded_abs": patched_action,
                "patched_value_to_recorded_abs": patched_value,
                "world_residual_fraction": patched_world / max(native_world, 1e-12),
                "action_residual_fraction": patched_action / max(native_action, 1e-12),
                "value_residual_fraction": patched_value / max(native_value, 1e-12),
            })
        rows.append({
            "episode": item["episode"], "frame": item["frame"],
            "task_a": item["task_a"], "task_b": item["task_b"],
            "native_world_goal_delta_abs": native_world,
            "native_action_goal_delta_abs": native_action,
            "native_value_goal_delta_abs": native_value,
            "layers": layer_results,
        })
        print(json.dumps({"episode": item["episode"], "native_world": native_world,
                          "native_action": native_action, "native_value": native_value}), flush=True)
    del model, vae
    gc.collect()
    torch.cuda.empty_cache()

    aggregate = {}
    for layer in LAYERS:
        vals = [x["world_residual_fraction"] for row in rows for x in row["layers"] if x["layer"] == layer]
        act = [x["action_residual_fraction"] for row in rows for x in row["layers"] if x["layer"] == layer]
        val = [x["value_residual_fraction"] for row in rows for x in row["layers"] if x["layer"] == layer]
        aggregate[str(layer)] = {
            "world_residual_fraction_mean": float(np.mean(vals)),
            "action_residual_fraction_mean": float(np.mean(act)),
            "value_residual_fraction_mean": float(np.mean(val)),
            "world_moves_toward_recorded_all": bool(all(x < 1.0 for x in vals)),
        }
    result = {
        "status": "completed_alphabrain_action_carrier_activation_mediation",
        "checkpoint": "models/AlphaBrain/cosmos_policy_libero/model.safetensors",
        "windows": len(rows), "episodes": [r["episode"] for r in rows],
        "layers": LAYERS, "patched_temporal_slots": PATCH_SLOTS,
        "world_slots": [5, 6, 7], "policy_slots": [4, 8],
        "sampler": "CosmosPolicySampler two-step reverse-ODE",
        "aggregate_by_layer": aggregate, "records": rows,
        "scope": "Internal paired activation patch on fixed real LIBERO image/state/recorded-action windows; no training, environment rollout, or task-harm claim.",
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "windows": result["windows"],
                      "aggregate_by_layer": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
