#!/usr/bin/env python3
"""Inference-only CPU weight residency with unchanged CUDA bf16 computations.

Exactly one sequential DiT block is on CUDA at a time. The VAE core visits CUDA
only during encode/decode. This changes weight placement, not model values,
quantization, conditioning, kernels, or action selection. No external process
is touched. It is an explicit execution option for the official adapter.
"""
from __future__ import annotations

import time

import torch


def tensor_bytes(module):
    return sum(t.numel() * t.element_size()
               for t in list(module.parameters()) + list(module.buffers()))


def move_direct_state(module, device):
    for parameter in module.parameters(recurse=False):
        parameter.data = parameter.data.to(device)
    for name, buffer in module.named_buffers(recurse=False):
        module._buffers[name] = buffer.to(device)


class OffloadedVAE(torch.nn.Module):
    def __init__(self, vae, device):
        super().__init__()
        self.base = vae
        self.device = device
        self.base.vae.to("cpu")
        # Keep only the tiny normalization buffers on CUDA so callers construct
        # their video/proprio inputs on the correct device.
        move_direct_state(self.base, device)

    @property
    def latents_mean(self):
        return self.base.latents_mean

    def _execute(self, method, value):
        if torch.is_grad_enabled():
            raise RuntimeError("Weight offloading is supported only under inference/no_grad")
        try:
            self.base.vae.to(self.device)
            return getattr(self.base, method)(value)
        finally:
            self.base.vae.to("cpu")
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode(self, value):
        return self._execute("encode", value)

    @torch.inference_mode()
    def decode(self, value):
        return self._execute("decode", value)


class LayerwiseOffload:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("This utility is specifically for CUDA computation")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("Load the complete DiT on CPU before installing offload")
        if not isinstance(model.blocks, torch.nn.ModuleList):
            raise TypeError("Expected the native sequential blocks ModuleList")
        self.handles = []
        self.block_count = len(model.blocks)
        self.block_bytes = [tensor_bytes(block) for block in model.blocks]
        self.total_bytes = tensor_bytes(model)
        self.fixed_bytes = self.total_bytes - sum(self.block_bytes)
        move_direct_state(model, self.device)
        for name, child in model.named_children():
            if name != "blocks":
                child.to(self.device)

    def _before(self, module, args):
        if torch.is_grad_enabled():
            raise RuntimeError("Weight offloading must run under inference/no_grad")
        try:
            module.to(self.device)
        except BaseException:
            module.to("cpu")
            torch.cuda.empty_cache()
            raise

    def _after(self, module, args, output):
        module.to("cpu")
        torch.cuda.empty_cache()

    def install(self):
        if self.handles:
            raise RuntimeError("Offload hooks are already installed")
        for block in self.model.blocks:
            self.handles.append(block.register_forward_pre_hook(self._before))
            self.handles.append(block.register_forward_hook(self._after, always_call=True))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.inference_mode()
    def check_first_block(self):
        """Exact CUDA output comparison before/after offload with real token shape."""
        if self.handles:
            raise RuntimeError("Validate before installing all hooks")
        started = time.monotonic()
        model = self.model
        generator = torch.Generator(device="cpu").manual_seed(471)
        x = torch.randn(1, 17, 9, 28, 28, generator=generator).to(self.device, torch.bfloat16)
        padding = torch.zeros(1, 1, 224, 224, device=self.device, dtype=torch.bfloat16)
        fps = torch.full((1,), 16, device=self.device, dtype=torch.bfloat16)
        embedded, rope, extra = model.prepare_embedded_sequence(x, fps=fps, padding_mask=padding)
        times = torch.full((1, 9), 80.0 / 81.0, device=self.device, dtype=torch.bfloat16)
        times[:, :4] = 0
        temb, adaln = model.t_embedder(times)
        temb = model.t_embedding_norm(temb)
        text = torch.randn(1, 512, 1024, generator=generator).to(self.device, torch.bfloat16)
        kwargs = {"rope_emb_L_1_1_D": rope, "adaln_lora_B_T_3D": adaln,
                  "extra_per_block_pos_emb": extra}
        block = model.blocks[0]
        before = after = None
        handles = []
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            block.to(self.device)
            before = block(embedded, temb, text, **kwargs).clone()
            block.to("cpu")
            torch.cuda.empty_cache()
            handles = [block.register_forward_pre_hook(self._before),
                       block.register_forward_hook(self._after, always_call=True)]
            after = block(embedded, temb, text, **kwargs)
            if not torch.equal(before, after):
                raise RuntimeError("Offload changed fixed-input block output")
            if any(p.device.type != "cpu" for p in block.parameters()):
                raise RuntimeError("Block weights did not return to CPU")
            return {"status": "passed", "exact_equal": True, "shape": list(after.shape),
                    "dtype": str(after.dtype), "max_abs_difference": 0.0,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                    "elapsed_seconds": time.monotonic() - started,
                    "scope": "First DiT block, same CUDA backend and bf16 inputs; no full-runtime or task guarantee"}
        finally:
            for handle in handles:
                handle.remove()
            block.to("cpu")
            del before, after
            torch.cuda.empty_cache()

    def metadata(self):
        return {"mode": "sequential CUDA blocks with CPU parameter residency",
                "precision_change": False, "quantization": False,
                "dit_blocks": self.block_count, "dit_total_tensor_bytes": self.total_bytes,
                "dit_resident_nonblock_bytes": self.fixed_bytes,
                "dit_max_block_bytes": max(self.block_bytes),
                "vae": "CUDA only for encode/decode, CPU otherwise"}


def prepare(model, vae, device="cuda", validate=True):
    manager = LayerwiseOffload(model, device=device)
    metadata = manager.metadata()
    if validate:
        metadata["fixed_input_block_parity"] = manager.check_first_block()
    manager.install()
    return model, OffloadedVAE(vae, manager.device), manager, metadata
