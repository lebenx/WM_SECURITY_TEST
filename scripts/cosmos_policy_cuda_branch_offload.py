#!/usr/bin/env python3
"""Keep only the called official VAE branch on CUDA, without changing its math.

The original wrapper and all model/codec sources remain unchanged. Encoding uses
encoder + conv1; decoding uses conv2 + decoder. Native cache initialization still
runs unchanged, including inspection of the inactive branch's module structure.
"""
from __future__ import annotations

import torch

from cosmos_policy_cuda_offload import LayerwiseOffload, move_direct_state, tensor_bytes


BRANCHES = {"encode": ("encoder", "conv1"), "decode": ("conv2", "decoder")}


def branch_audit(vae):
    core = vae.vae
    names = tuple(name for name, _ in core.named_children())
    expected = ("encoder", "conv1", "conv2", "decoder")
    if names != expected:
        raise TypeError(f"Unverified VAE structure: {names}")
    if list(core.parameters(recurse=False)) or list(core.buffers(recurse=False)):
        raise TypeError("Unverified state outside the four VAE branches")
    if any(t.dtype != torch.bfloat16 for t in core.parameters()):
        raise TypeError("This adapter preserves the official BF16 VAE parameters")
    children = {name: {"parameters": sum(p.numel() for p in child.parameters()),
                       "tensor_bytes": tensor_bytes(child)}
                for name, child in core.named_children()}
    return {"children": children, "core_tensor_bytes": tensor_bytes(core),
            "direct_wrapper_tensor_bytes": sum(t.numel() * t.element_size()
                for t in list(vae.parameters(recurse=False)) + list(vae.buffers(recurse=False))),
            "branch_tensor_bytes": {method: sum(children[name]["tensor_bytes"] for name in names)
                                    for method, names in BRANCHES.items()},
            "temporal_window": core.temporal_window,
            "precision": "torch.bfloat16", "precision_change": False,
            "quantization": False, "native_encode_decode_implementation_unchanged": True}


class BranchOffloadedVAE(torch.nn.Module):
    def __init__(self, vae, device="cuda"):
        super().__init__()
        self.audit = branch_audit(vae)
        self.base = vae
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("This utility is specifically for CUDA computation")
        self.base.vae.to("cpu")
        move_direct_state(self.base, self.device)
        self._active = False

    @property
    def latents_mean(self):
        return self.base.latents_mean

    def _execute(self, method, value):
        if torch.is_grad_enabled():
            raise RuntimeError("Branch residency is inference/no_grad only")
        if self._active:
            raise RuntimeError("A branch-offloaded VAE cannot serve concurrent calls")
        if any(t.device.type != "cpu" for t in self.base.vae.parameters()):
            raise RuntimeError("VAE core must be CPU resident between calls")
        branches = [getattr(self.base.vae, name) for name in BRANCHES[method]]
        self._active = True
        try:
            for module in branches:
                module.to(self.device)
            return getattr(self.base, method)(value)
        finally:
            # Native calls clear these caches themselves on success. Also clear
            # after an interrupted/OOM call so hidden activation tensors release.
            self.base.vae.clear_cache()
            for module in branches:
                module.to("cpu")
            self._active = False
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode(self, value):
        return self._execute("encode", value)

    @torch.inference_mode()
    def decode(self, value):
        return self._execute("decode", value)


def prepare(model, vae, device="cuda", validate=True):
    """Drop-in execution adapter; caller must separately record this source SHA."""
    manager = LayerwiseOffload(model, device=device)
    metadata = manager.metadata()
    if validate:
        metadata["fixed_input_block_parity"] = manager.check_first_block()
    manager.install()
    wrapped = BranchOffloadedVAE(vae, manager.device)
    metadata["vae"] = "CUDA encoder+conv1 OR conv2+decoder only; CPU core between calls"
    metadata["vae_branch_residency"] = wrapped.audit
    return model, wrapped, manager, metadata
