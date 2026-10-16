#!/usr/bin/env python3
"""CPU checks against released methods and checkpoint; never uses a GPU."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

import alphabrain_checkpoint_inference as exact
import pilot_alphabrain_cosmos_libero_short_control as legacy


def released_methods():
    path = exact.ROOT / "vendor/AlphaBrain/AlphaBrain/model/framework/CosmosPolicy.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CosmosPolicy")
    methods = {"_edm_scaling", "_denoise", "_preprocess_image", "_extract_action_from_latent"}
    selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class CaptureDiT:
    use_wan_fp32_strategy = False

    def __call__(self, **kwargs):
        self.inputs = {key: value.clone() for key, value in kwargs.items()}
        return kwargs["x_B_C_T_H_W"] * 0.25 + kwargs["timesteps_B_T"][:, None, :, None, None]


def main():
    torch.set_num_threads(2)
    methods = released_methods()
    model = CaptureDiT()
    reference = SimpleNamespace(sigma_data=1.0, sigma_conditional=0.0, dit=model)
    reference._edm_scaling = lambda sigma: methods["_edm_scaling"](reference, sigma)
    generator = torch.Generator().manual_seed(90210)
    x = torch.randn(2, 16, 9, 4, 4, generator=generator)
    gt = torch.randn(2, 16, 4, 4, 4, generator=generator).to(torch.bfloat16)
    text = torch.randn(2, 3, 1024, generator=generator)
    padding = torch.zeros(2, 1, 32, 32)
    fps = torch.tensor([16, 16])
    for sigma in (4.0, 20.0, 80.0):
        noise = torch.tensor([sigma, sigma])
        expected = methods["_denoise"](reference, x, noise, text, fps, padding, gt, 4)
        expected_inputs = model.inputs
        actual = exact.denoise(model, x, noise, text, fps, padding, gt)
        assert torch.equal(actual, expected), f"Denoise output mismatch at {sigma}"
        for key in expected_inputs:
            assert torch.equal(model.inputs[key], expected_inputs[key]), f"DiT input mismatch: {key}"
        assert torch.count_nonzero(model.inputs["timesteps_B_T"][:, :4]) == 0
    image = np.random.default_rng(13).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert np.array_equal(legacy._official_preprocess_image(image), methods["_preprocess_image"](None, image))
    action_latent = torch.randn(2, 16, 28, 28, generator=generator)
    action = methods["_extract_action_from_latent"](SimpleNamespace(chunk_size=16, action_dim=7), action_latent)
    assert torch.equal(action, legacy.action_from_latent(action_latent))
    stats = json.loads((exact.ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_dataset_statistics.json").read_text())
    raw = torch.tensor([[[-3., 3., 0.1, 3., -3., 0.2, 0.0]]])
    decoded = exact.normalize_action(raw, stats)
    low, high = (np.array(stats[key], dtype=np.float32) for key in ("actions_min", "actions_max"))
    assert (decoded >= low).all() and (decoded <= high).all()
    vae = exact.CheckpointWanVAE(device="cpu")
    state = vae.state_dict()
    with safe_open(str(exact.MODEL), framework="pt", device="cpu") as handle:
        assert sum(key.startswith("vae.") for key in handle.keys()) == len(state)
        for key, tensor in state.items():
            assert torch.equal(tensor, handle.get_tensor("vae." + key)), key
    video = torch.randn(1, 3, 33, 32, 32, generator=generator).clamp(-1, 1).to(torch.bfloat16)
    encoded = vae.encode(video)
    assert encoded.shape == (1, 16, 9, 4, 4)
    assert torch.isfinite(encoded).all()
    print(json.dumps({
        "status": "passed", "device": "cpu", "checkpoint_vae_tensor_equality": len(state),
        "denoise_exact_reference_parity_sigma": [4, 20, 80],
        "denoise_compared": "output and every DiT input against released CosmosPolicy._denoise",
        "image_preprocessing_reference_exact": True, "action_extraction_reference_exact": True,
        "inverse_scaling_bounds_passed": True,
        "native_vae_finite_encode_shape": list(encoded.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
