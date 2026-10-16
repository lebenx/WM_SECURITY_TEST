#!/usr/bin/env python3
"""Fixed-noise goal-swap probe for the released AlphaBrain CosmosPolicy.

The probe calls the released DiT directly on a fixed latent state.  It does
not encode images, execute LIBERO, or claim a task result.  Its only purpose is
to test whether changing the supported T5 task embedding changes the latent
slots that the model documents as future observations.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors"
TEXT = ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_t5_embeddings.pkl"
OUT = ROOT / "runs/alphabrain_cosmos_policy_goal_swap_v0/summary.json"


def install_transformer_engine_stub() -> None:
    """Use PyTorch RMSNorm and a faithful simple 3D RoPE for this audit host."""
    te = types.ModuleType("transformer_engine")
    te.__version__ = "2.7.0"
    pyt = types.ModuleType("transformer_engine.pytorch")
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (y * self.weight.float()).to(dtype=x.dtype)

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)

    pyt.RMSNorm = RMSNorm
    attn = types.ModuleType("transformer_engine.pytorch.attention")

    def rope(x, emb, tensor_format="bshd", fused=True):
        del tensor_format, fused
        # Source creates angle embeddings as (L,1,1,D); q/k are (B,L,H,D).
        if emb.ndim == 4:
            emb = emb.permute(1, 0, 2, 3)
        d = x.shape[-1]
        emb = emb[..., :d].to(dtype=x.dtype, device=x.device)
        cos, sin = torch.cos(emb), torch.sin(emb)
        x1, x2 = x[..., : d // 2], x[..., d // 2 : 2 * (d // 2)]
        rot = torch.cat([-x2, x1], dim=-1)
        if d % 2:
            rot = torch.cat([rot, x[..., -1:]], dim=-1)
        return x * cos + rot * sin

    attn.apply_rotary_pos_emb = rope
    pyt.attention = attn
    te.pytorch = pyt
    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = pyt
    sys.modules["transformer_engine.pytorch.attention"] = attn

    # The released source imports FSDP composable helpers even for single-GPU
    # inference.  Torch 2.2 (the local WorldAgen environment) predates them.
    fsdp = types.ModuleType("torch.distributed._composable.fsdp")
    fsdp.fully_shard = lambda module, **kwargs: module
    sys.modules["torch.distributed._composable.fsdp"] = fsdp

    flex_pkg = types.ModuleType("torch.nn.attention")
    class SDPBackend:
        MATH = "math"
        FLASH_ATTENTION = "flash"
        EFFICIENT_ATTENTION = "efficient"

    class _SDPAContext:
        def __enter__(self): return self
        def __exit__(self, *args): return False

    flex_pkg.SDPBackend = SDPBackend
    flex_pkg.sdpa_kernel = lambda *args, **kwargs: _SDPAContext()
    flex_mod = types.ModuleType("torch.nn.attention.flex_attention")
    flex_mod.BlockMask = object
    flex_mod.create_block_mask = lambda *args, **kwargs: None
    flex_mod.flex_attention = lambda *args, **kwargs: None
    sys.modules["torch.nn.attention"] = flex_pkg
    sys.modules["torch.nn.attention.flex_attention"] = flex_mod


class MaskedSelfAttention(nn.Module):
    """Preserve the stock self-attention except for selected row/key edges."""

    def __init__(self, original, blocked_rows, blocked_cols):
        super().__init__()
        self.original = original
        self.blocked_rows = blocked_rows
        self.blocked_cols = blocked_cols

    def forward(self, x, context=None, rope_emb=None, **kwargs):
        del context, kwargs
        q, k, v = self.original.compute_qkv(x, None, rope_emb=rope_emb)
        # Attention computes over (B,H,L,D); the source uses a full, unmasked
        # self-attention kernel for this backend.
        qh, kh, vh = (z.permute(0, 2, 1, 3) for z in (q, k, v))
        L = q.shape[1]
        mask = torch.zeros((L, L), dtype=qh.dtype, device=qh.device)
        rows = self.blocked_rows if isinstance(self.blocked_rows, (list, tuple)) else [self.blocked_rows]
        cols = self.blocked_cols if isinstance(self.blocked_cols, (list, tuple)) else [self.blocked_cols]
        for row in rows:
            for col in cols:
                mask[row, col] = float("-inf")
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, dropout_p=0.0)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], L, -1)
        return self.original.output_proj(out)


class GoalCutCrossAttention(nn.Module):
    """Keep task context for policy slots, cut it for future-world slots."""

    def __init__(self, original, blocked_rows):
        super().__init__()
        self.original = original
        self.blocked_rows = blocked_rows

    def forward(self, x, context=None, rope_emb=None, **kwargs):
        out = self.original(x, context, rope_emb=rope_emb, **kwargs)
        out = out.clone()
        rows = self.blocked_rows if isinstance(self.blocked_rows, (list, tuple)) else [self.blocked_rows]
        for row in rows:
            out[:, row, :] = 0
        return out


def install_goal_bus_cut(model, tokens_per_slot: int = 196, *, cut_self_attention=True) -> None:
    """Cut text inputs and optionally policy/value -> observation edges."""
    condition = slice(0, 4 * tokens_per_slot)
    world = slice(5 * tokens_per_slot, 8 * tokens_per_slot)
    policy = slice(4 * tokens_per_slot, 5 * tokens_per_slot)
    value = slice(8 * tokens_per_slot, 9 * tokens_per_slot)
    goal_free_rows = [condition, world]
    goal_bearing_cols = [policy, value]
    for block in model.blocks:
        # Only action/value rows retain instruction cross-attention.  All
        # observation rows are also prevented from reading those two rows.
        if not isinstance(block.cross_attn, GoalCutCrossAttention):
            block.cross_attn = GoalCutCrossAttention(block.cross_attn, goal_free_rows)
        if cut_self_attention:
            block.self_attn = MaskedSelfAttention(block.self_attn, goal_free_rows, goal_bearing_cols)


def install_world_text_only_cut(model, tokens_per_slot: int = 196) -> None:
    """Remove text from world/value rows while retaining the action carrier.

    Unlike ``install_goal_bus_cut``, this refined fixed-action intervention
    leaves self-attention intact. The sampler-held action slot can therefore
    still influence world rows; value is made text-free so it cannot carry the
    removed goal signal back into world predictions.
    """
    condition = slice(0, 4 * tokens_per_slot)
    world = slice(5 * tokens_per_slot, 8 * tokens_per_slot)
    value = slice(8 * tokens_per_slot, 9 * tokens_per_slot)
    blocked_rows = [condition, world, value]
    for block in model.blocks:
        block.cross_attn = GoalCutCrossAttention(block.cross_attn, [condition, world, value])


def install_fixed_action_world_cut(model, tokens_per_slot: int = 196) -> None:
    """Goal-free world branch with a fixed-action carrier.

    Text cross-attention is removed from condition, action, world, and value
    rows, but self-attention remains unmasked. A sampler-held action row can
    therefore affect world rows while no intermediate carrier can reintroduce
    task text. Native policy outputs should be read from a parallel uncut
    branch; this function repairs the world branch only.
    """
    condition = slice(0, 4 * tokens_per_slot)
    policy = slice(4 * tokens_per_slot, 5 * tokens_per_slot)
    world = slice(5 * tokens_per_slot, 8 * tokens_per_slot)
    value = slice(8 * tokens_per_slot, 9 * tokens_per_slot)
    blocked_rows = [condition, policy, world, value]
    for block in model.blocks:
        block.cross_attn = GoalCutCrossAttention(block.cross_attn, blocked_rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this short probe")
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        install_transformer_engine_stub()
    sys.path.insert(0, str(ROOT / "vendor/AlphaBrain"))
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT

    model = MinimalV1LVGDiT(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True, model_channels=2048, num_blocks=28,
        num_heads=16, crossattn_emb_channels=1024, use_crossattn_projection=False,
        atten_backend="torch", pos_emb_cls="rope3d", mlp_ratio=4.0,
        use_adaln_lora=True, adaln_lora_dim=256, use_wan_fp32_strategy=False,
        pos_emb_learnable=True, rope_enable_fps_modulation=False,
        rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
    ).cuda().to(dtype=torch.bfloat16).eval()
    sd = load_file(str(MODEL), device="cpu")
    sd = {k.removeprefix("dit."): v for k, v in sd.items() if k.startswith("dit.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"state mismatch missing={len(missing)} unexpected={len(unexpected)} first_unexpected={unexpected[:8]}")
    del sd

    text = pickle.load(open(TEXT, "rb"))
    names = list(text)
    if len(names) < 2:
        raise RuntimeError("need at least two released task embeddings")
    pairs = [(names[0], names[1]), (names[2], names[3]), (names[4], names[5])]
    x = torch.Generator(device="cuda").manual_seed(19)
    latent = torch.randn((1, 16, 9, 28, 28), generator=x, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    cond = torch.zeros((1, 1, 9, 28, 28), device="cuda", dtype=torch.bfloat16)
    cond[:, :, :4] = 1
    timestep = torch.full((1, 9), 0.45, device="cuda", dtype=torch.bfloat16)
    fps = torch.tensor([16], device="cuda", dtype=torch.bfloat16)
    padding = torch.zeros((1, 1, 224, 224), device="cuda", dtype=torch.bfloat16)
    pair_results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a, b in pairs:
            ctx_a = text[a].cuda().to(dtype=torch.bfloat16)
            ctx_b = text[b].cuda().to(dtype=torch.bfloat16)
            ya = model(latent, timestep, ctx_a, cond, fps=fps, padding_mask=padding)
            yb = model(latent, timestep, ctx_b, cond, fps=fps, padding_mask=padding)
            delta = (ya.float() - yb.float()).abs().mean(dim=(0, 1, 3, 4)).cpu()
            rel = ((ya.float() - yb.float()).pow(2).mean().sqrt() / ya.float().pow(2).mean().sqrt()).item()
            pair_results.append({
                "task_a": a,
                "task_b": b,
                "per_slot_mean_abs_delta": [float(v) for v in delta],
                "future_wrist_delta": float(delta[6]),
                "future_primary_delta": float(delta[7]),
                "all_prediction_slots_nonzero": bool(torch.all(delta[4:] > 0)),
                "relative_output_l2": rel,
            })
        # A zero context is a diagnostic no-goal baseline.  It is not claimed
        # to be a released training condition; it only separates task-text
        # sensitivity from the arbitrary choice of two legal task strings.
        ctx_zero = torch.zeros_like(text[pairs[0][0]].cuda().to(dtype=torch.bfloat16))
        y_zero = model(latent, timestep, ctx_zero, cond, fps=fps, padding_mask=padding)
        y_a = model(latent, timestep, text[pairs[0][0]].cuda().to(dtype=torch.bfloat16), cond, fps=fps, padding_mask=padding)
        no_goal_delta = (y_a.float() - y_zero.float()).abs().mean(dim=(0, 1, 3, 4)).cpu()
        # Direct text isolation alone leaves contextualized action/value slots
        # readable.  Compare against the full cut on the same pair and input.
        install_goal_bus_cut(model, cut_self_attention=False)
        ya_direct = model(latent, timestep, text[pairs[0][0]].cuda().to(dtype=torch.bfloat16), cond, fps=fps, padding_mask=padding)
        yb_direct = model(latent, timestep, text[pairs[0][1]].cuda().to(dtype=torch.bfloat16), cond, fps=fps, padding_mask=padding)
        direct_delta = (ya_direct.float() - yb_direct.float()).abs().mean(dim=(0, 1, 3, 4)).cpu()
        # One matched repair pair: cut text from future-world slots and block
        # their self-attention reads from action/value slots, while policy rows
        # retain the original task context.
        install_goal_bus_cut(model)
        a, b = pairs[0]
        ya_cut = model(latent, timestep, text[a].cuda().to(dtype=torch.bfloat16), cond, fps=fps, padding_mask=padding)
        yb_cut = model(latent, timestep, text[b].cuda().to(dtype=torch.bfloat16), cond, fps=fps, padding_mask=padding)
        cut_delta = (ya_cut.float() - yb_cut.float()).abs().mean(dim=(0, 1, 3, 4)).cpu()
        cut_rel = ((ya_cut.float() - yb_cut.float()).pow(2).mean().sqrt() / ya_cut.float().pow(2).mean().sqrt()).item()
        y_zero_cut = model(latent, timestep, ctx_zero, cond, fps=fps, padding_mask=padding)
        no_goal_cut_delta = (ya_cut.float() - y_zero_cut.float()).abs().mean(dim=(0, 1, 3, 4)).cpu()
    result = {
        "status": "completed_alphabrain_fixed_latent_goal_swap",
        "runtime_backend": "native_transformer_engine" if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") == "1" else "local_transformer_engine_compatibility_stub",
        "checkpoint": str(MODEL.relative_to(ROOT)),
        "task_pairs": pair_results,
        "fixed_latent_shape": list(latent.shape),
        "condition_slots_fixed": [0, 1, 2, 3],
        "prediction_slots": {"action": 4, "future_proprio": 5, "future_wrist": 6, "future_primary": 7, "value": 8},
        "all_pairs_prediction_slots_nonzero": all(x["all_prediction_slots_nonzero"] for x in pair_results),
        "direct_text_cut_only_pair": {
            "task_a": pairs[0][0],
            "task_b": pairs[0][1],
            "per_slot_mean_abs_delta": [float(v) for v in direct_delta],
            "future_world_delta_mean": float(direct_delta[5:8].mean()),
            "all_future_world_slots_nonzero": bool(torch.all(direct_delta[5:8] > 0)),
            "policy_delta_mean": float(direct_delta[[4, 8]].mean()),
        },
        "no_goal_zero_context_diagnostic": {
            "native_per_slot_mean_abs_delta": [float(v) for v in no_goal_delta],
            "native_future_world_delta_mean": float(no_goal_delta[5:8].mean()),
            "native_policy_delta_mean": float(no_goal_delta[[4, 8]].mean()),
            "repair_per_slot_mean_abs_delta": [float(v) for v in no_goal_cut_delta],
            "repair_future_world_delta_max": float(no_goal_cut_delta[5:8].max()),
            "repair_policy_delta_mean": float(no_goal_cut_delta[[4, 8]].mean()),
            "repair_future_world_exact_zero": bool(torch.all(no_goal_cut_delta[5:8] == 0)),
            "interpretation": "diagnostic zero T5 context; not a released training condition",
        },
        "matched_repair_pair": {
            "task_a": pairs[0][0],
            "task_b": pairs[0][1],
            "cut_world_slots": [5, 6, 7],
            "cut_policy_slots": [4, 8],
            "per_slot_mean_abs_delta": [float(v) for v in cut_delta],
            "future_world_delta_max": float(cut_delta[5:8].max()),
            "policy_delta_mean": float(cut_delta[[4, 8]].mean()),
            "future_world_exact_zero": bool(torch.all(cut_delta[5:8] == 0)),
            "relative_output_l2": cut_rel,
        },
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "scope": "fixed latent one-step DiT output with a matched row-wise repair; no image decode, environment step, or task-level harm",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
