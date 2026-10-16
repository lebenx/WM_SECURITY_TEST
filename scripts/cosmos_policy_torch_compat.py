#!/usr/bin/env python3
"""Torch-only compatibility for the official Cosmos adapter, not full TE parity.

RoPE follows NVIDIA Transformer Engine v2.7's unfused precision order: evaluate
trigonometric functions on FP32 angles, then cast their values to the input
dtype. The earlier Alpha diagnostic shim is deliberately left unchanged.
CUDA fused TE kernels, RMSNorm kernel parity, and full-model parity remain
outside the CPU component test. Only the native BSHD/SBHD, noninterleaved,
single-process RoPE contract is supported here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Optional

import torch
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[1]
TE_SOURCE_URL = "https://raw.githubusercontent.com/NVIDIA/TransformerEngine/v2.7/transformer_engine/pytorch/attention/rope.py"
TE_SOURCE_SHA256 = "f96ac3450e19a3c248ba74e9ee22752ee3704d7b40be67b12884230b5362da01"
TE_SOURCE = ROOT / "runs/alphabrain_official_forecast_contract_audit_v1/sources/transformer_engine_v2_7_rope.py"


def apply_rotary_pos_emb(t, freqs, tensor_format="bshd", fused=True):
    """The official unfused arithmetic for the Cosmos native RoPE call.

    ``fused=True`` is accepted because the DiT passes it, but execution remains
    unfused Torch. Do not advertise fused CUDA numerical equivalence.
    """
    del fused
    if tensor_format not in ("bshd", "sbhd"):
        raise ValueError("Only native bshd/sbhd RoPE is supported")
    if freqs.ndim != 4 or t.ndim != 4 or freqs.dtype != torch.float32:
        raise ValueError("RoPE requires four-dimensional input and FP32 angle tensor")
    sequence_length = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    if sequence_length > freqs.shape[0]:
        raise ValueError("RoPE angle sequence is shorter than the input")
    rotary_dim = freqs.shape[-1]
    if rotary_dim > t.shape[-1] or rotary_dim % 2:
        raise ValueError("RoPE dimension must be even and no larger than head dimension")
    freqs = freqs[:sequence_length]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    # Critical: never quantize angles to bf16 before evaluating cos/sin.
    cos = torch.cos(freqs).to(t.dtype)
    sin = torch.sin(freqs).to(t.dtype)
    active, passthrough = t[..., :rotary_dim], t[..., rotary_dim:]
    first, second = torch.chunk(active, 2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((active * cos + rotated * sin, passthrough), dim=-1)


def compatibility_metadata():
    return {"backend": "torch compatibility; official TE unfused RoPE arithmetic",
            "shim_path": str(Path(__file__).resolve()),
            "shim_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rope_reference_url": TE_SOURCE_URL, "rope_reference_sha256": TE_SOURCE_SHA256,
            "rope_angle_precision": "FP32 cos/sin then cast to q/k dtype",
            "full_transformer_engine_runtime_parity": False,
            "cuda_fused_rope_parity": False,
            "rmsnorm_fused_kernel_parity": False}


def _extract_node(node, label, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                                           names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, label, "exec"), namespace)


def self_test(reference=TE_SOURCE):
    """Execute pinned official functions against native-size CPU inputs."""
    reference = Path(reference)
    raw = reference.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TE_SOURCE_SHA256:
        raise ValueError("Official TE reference source SHA256 differs")
    namespace = {"torch": torch, "nn": torch.nn, "Optional": Optional,
                 "repeat": repeat, "rearrange": rearrange}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_rotate_half", "_apply_rotary_pos_emb_base"):
            _extract_node(node, str(reference), namespace)
    dit_path = ROOT / "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/official_dit/minimal_v4_dit.py"
    for node in ast.parse(dit_path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in ("VideoPositionEmb", "VideoRopePosition3DEmb"):
            _extract_node(node, str(dit_path), namespace)
    position = namespace["VideoRopePosition3DEmb"](
        head_dim=128, len_h=120, len_w=120, len_t=128,
        h_extrapolation_ratio=3, w_extrapolation_ratio=3,
        enable_fps_modulation=False).to(torch.bfloat16)
    freqs = position.generate_embeddings(torch.Size((1, 9, 14, 14, 2048)))
    generator = torch.Generator(device="cpu").manual_seed(471)
    original = torch.randn(1, 1764, 16, 128, generator=generator).bfloat16()
    rows = []
    for tensor_format in ("bshd", "sbhd"):
        for dtype in (torch.bfloat16, torch.float32):
            x = original.to(dtype)
            if tensor_format == "sbhd":
                x = x.transpose(0, 1)
            expected = namespace["_apply_rotary_pos_emb_base"](x, freqs, tensor_format=tensor_format)
            actual = apply_rotary_pos_emb(x, freqs, tensor_format=tensor_format)
            if not torch.equal(actual, expected):
                raise AssertionError("New RoPE differs from official unfused reference")
            rows.append({"tensor_format": tensor_format, "dtype": str(dtype),
                         "shape": list(x.shape), "exact_equal": True, "max_abs_difference": 0.0})
    # Preserve the prior component discrepancy as measured evidence, without
    # rewriting the original Alpha shim or claiming task-level consequences.
    old_path = ROOT / "scripts/pilot_alphabrain_cosmos_policy_goal_swap.py"
    for node in ast.walk(ast.parse(old_path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "rope":
            _extract_node(node, str(old_path), namespace)
    old = namespace["rope"](original, freqs, tensor_format="bshd")
    corrected = apply_rotary_pos_emb(original, freqs, tensor_format="bshd")
    delta = old.float() - corrected.float()
    return {"status": "passed", "device": "cpu", "seed": 471,
            "model_weight_loads": 0, "learned_model_forward_calls": 0,
            "metadata": compatibility_metadata(),
            "official_reference_path": str(reference.resolve()),
            "position_source_sha256": hashlib.sha256(dit_path.read_bytes()).hexdigest(),
            "position_shape": [1, 9, 14, 14, 2048], "frequency_dtype": str(freqs.dtype),
            "official_unfused_exact_parity": rows,
            "legacy_shim_difference": {
                "legacy_source_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                "changed_elements": int(torch.count_nonzero(delta)), "total_elements": delta.numel(),
                "fraction_changed": float(torch.count_nonzero(delta) / delta.numel()),
                "max_abs_difference": float(delta.abs().max()),
                "relative_l2": float(delta.norm() / corrected.float().norm()),
                "task_outcome_measured": False},
            "limits": "CPU RoPE arithmetic only; not full TE/fused CUDA/DiT/task parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference", type=Path, default=TE_SOURCE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; imports do not install the compatibility module")
    torch.set_num_threads(2)
    result = self_test(args.reference)
    if args.out:
        with args.out.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def install_transformer_engine_stub() -> None:
    """Install a Torch-only compatibility module with official unfused RoPE precision ordering."""
    loaded_dit = sys.modules.get("AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v4_dit")
    if loaded_dit is not None and loaded_dit.apply_rotary_pos_emb is not apply_rotary_pos_emb:
        raise RuntimeError("A different RoPE backend is already imported; start a fresh official-adapter process")
    te = types.ModuleType('transformer_engine')
    te.__version__ = '2.7.0'
    pyt = types.ModuleType('transformer_engine.pytorch')

    class RMSNorm(torch.nn.Module):

        def __init__(self, dim, eps=1e-06):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (y * self.weight.float()).to(dtype=x.dtype)

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)
    pyt.RMSNorm = RMSNorm
    attn = types.ModuleType('transformer_engine.pytorch.attention')
    attn.apply_rotary_pos_emb = apply_rotary_pos_emb
    pyt.attention = attn
    te.pytorch = pyt
    sys.modules['transformer_engine'] = te
    sys.modules['transformer_engine.pytorch'] = pyt
    sys.modules['transformer_engine.pytorch.attention'] = attn
    fsdp = types.ModuleType('torch.distributed._composable.fsdp')
    fsdp.fully_shard = lambda module, **kwargs: module
    sys.modules['torch.distributed._composable.fsdp'] = fsdp
    flex_pkg = types.ModuleType('torch.nn.attention')

    class SDPBackend:
        MATH = 'math'
        FLASH_ATTENTION = 'flash'
        EFFICIENT_ATTENTION = 'efficient'

    class _SDPAContext:

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False
    flex_pkg.SDPBackend = SDPBackend
    flex_pkg.sdpa_kernel = lambda *args, **kwargs: _SDPAContext()
    flex_mod = types.ModuleType('torch.nn.attention.flex_attention')
    flex_mod.BlockMask = object
    flex_mod.create_block_mask = lambda *args, **kwargs: None
    flex_mod.flex_attention = lambda *args, **kwargs: None
    sys.modules['torch.nn.attention'] = flex_pkg
    sys.modules['torch.nn.attention.flex_attention'] = flex_mod


if __name__ == "__main__":
    main()
