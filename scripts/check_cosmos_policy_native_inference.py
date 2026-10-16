#!/usr/bin/env python3
"""CPU-only parity checks for the pinned official Cosmos Policy adapter.

These are implementation checks, not real-task or attack results. Use
--weights after the pinned checkpoint and tokenizer downloads are complete.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange

import cosmos_policy_native_inference as native
import cosmos_policy_codecs as codecs
import cosmos_policy_wan_vae as vae_core

SOURCE = native.ROOT / "runs/alphabrain_official_forecast_contract_audit_v1/sources/cosmos_policy"


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class CaptureDiT:
    use_wan_fp32_strategy = False

    def __call__(self, **kwargs):
        self.inputs = {key: value.clone() for key, value in kwargs.items()}
        return kwargs["x_B_C_T_H_W"] * 0.25 + kwargs["timesteps_B_T"][:, None, :, None, None]


def source_parity():
    core_src = ast.parse((SOURCE / "_src/predict2/tokenizers/wan2pt1.py").read_text())
    core_dst = ast.parse(Path(vae_core.__file__).read_text())
    copied = [n for n in core_dst.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    for node in copied:
        original = next(n for n in core_src.body if isinstance(n, type(node)) and n.name == node.name)
        assert ast.dump(node) == ast.dump(original), node.name
    codec_sources = [ast.parse((SOURCE / p).read_text()) for p in (
        "datasets/dataset_utils.py", "experiments/robot/cosmos_utils.py", "models/policy_text2world_model.py")]
    copied_codecs = [n for n in ast.parse(Path(codecs.__file__).read_text()).body if isinstance(n, ast.FunctionDef)]
    for node in copied_codecs:
        original = next(n for t in codec_sources for n in t.body
                        if isinstance(n, ast.FunctionDef) and n.name == node.name)
        assert ast.dump(node) == ast.dump(original), node.name
    scaling = extract(SOURCE / "_src/imaginaire/modules/denoiser_scaling.py", "RectifiedFlowScaling", {"torch": torch})(1.0)
    denoise = extract(SOURCE / "models/policy_video2world_model.py", "denoise",
                      {"torch": torch, "rearrange": rearrange,
                       "DenoisePrediction": namedtuple("DenoisePrediction", "x0 eps logvar")},
                      "CosmosPolicyVideo2WorldModel")
    model = CaptureDiT()
    ref = SimpleNamespace(net=model, scaling=scaling,
                          config=SimpleNamespace(sigma_data=1.0, sigma_conditional=0.0,
                                                 use_wan_fp32_strategy=False, denoise_replace_gt_frames=True),
                          tensor_kwargs={"device": "cpu", "dtype": torch.bfloat16})
    rng = torch.Generator().manual_seed(8873)
    xt = torch.randn(2, 16, 9, 4, 4, generator=rng)
    gt = torch.randn(2, 16, 9, 4, 4, generator=rng).to(torch.bfloat16)
    text = torch.randn(2, 3, 1024, generator=rng).to(torch.bfloat16)
    fps = torch.full((2,), 16, dtype=torch.bfloat16)
    padding = torch.zeros(2, 1, 32, 32, dtype=torch.bfloat16)
    comparisons = []
    for n_cond in (4, 5, 8):
        mask = torch.zeros(2, 1, 9, 4, 4, dtype=torch.bfloat16)
        mask[:, :, :n_cond] = 1
        condition = SimpleNamespace(is_video=True, use_video_condition=True, gt_frames=gt,
                                    condition_video_input_mask_B_C_T_H_W=mask)
        condition.to_dict = lambda: {"crossattn_emb": text, "fps": fps, "padding_mask": padding,
                                      "condition_video_input_mask_B_C_T_H_W": mask}
        for sigma in (4., 20., 80.):
            s = torch.full((2,), sigma)
            expected = denoise(ref, xt, s, condition).x0
            inputs = model.inputs
            actual = native.alpha.denoise(model, xt, s, text, fps, padding, gt[:, :, :n_cond])
            assert torch.equal(expected, actual), (n_cond, sigma)
            assert inputs.keys() == model.inputs.keys()
            for key in inputs:
                assert torch.equal(inputs[key], model.inputs[key]), (n_cond, sigma, key)
            comparisons.append({"n_cond": n_cond, "sigma": sigma})
    source_rand = extract(SOURCE / "_src/imaginaire/utils/misc.py", "arch_invariant_rand",
                           {"torch": torch, "np": np})
    native.alpha._source_path()
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
    for seed in (1, 31, 32):
        assert torch.equal(source_rand((1, 16, 9, 4, 4), torch.float32, "cpu", seed),
                           arch_invariant_rand((1, 16, 9, 4, 4), seed=seed))
    # Test public helper stages preserve every fixed action-latent element.
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    sampler = CosmosPolicySampler()
    for n_cond, steps in ((4, 5), (5, 1), (8, 1)):
        sample = native.sample_conditioned(model, sampler, arch_invariant_rand,
                                           gt[:1], text[:1], n_cond, seed=31, num_steps=steps)
        assert torch.equal(sample[:, :, :n_cond], gt[:1, :, :n_cond].float())
    return {"vae_core_ast_identical_definitions": len(copied),
            "official_codecs_ast_identical_functions": len(copied_codecs),
            "official_denoise_all_inputs_and_outputs_exact": comparisons,
            "official_arch_invariant_noise_exact_seeds": [1, 31, 32],
            "native_prefix_clamp_checked": [4, 5, 8]}


def sampler_parity():
    native.alpha._source_path()
    from AlphaBrain.model.modules.world_model.cosmos import cosmos_sampler as local
    ns = {"torch": torch, "math": math, "log": SimpleNamespace(info=lambda *a: None),
          "SamplerConfig": local.SamplerConfig,
          "SolverConfig": local.SolverConfig, "SolverTimestampConfig": local.SolverTimestampConfig}
    for path in ("batch_ops.py", "runge_kutta.py", "multi_step.py"):
        source = SOURCE / "_src/imaginaire/functional" / path
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(source), "exec"), ns)
    source = SOURCE / "_src/imaginaire/modules/res_sampler.py"
    for name in ("get_rev_ts", "fori_loop", "differential_equation_solver", "Sampler"):
        extract(source, name, ns)
    reference = extract(SOURCE / "modules/cosmos_sampler.py", "CosmosPolicySampler", ns)()
    local_sampler = local.CosmosPolicySampler()
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(89)) * 80
    fn = lambda xt, sigma: torch.tanh(xt / (sigma[:, None, None] + 1.0))
    for steps in (1, 5):
        expected = reference(fn, x, num_steps=steps, sigma_min=4., sigma_max=80.)
        actual = local_sampler(fn, x, num_steps=steps, sigma_min=4., sigma_max=80.)
        assert torch.equal(expected, actual), steps
    return {"official_sampler_and_solver_output_exact_steps": [1, 5],
            "sampling_step_scope": "Only published action5/future1/value1 configuration supported; Alpha port differs from upstream for num_steps=2, excluded by adapter."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {"status": "passed", "device": "cpu", "source_revision": native.SOURCE_REVISION,
              "checks": {**source_parity(), **sampler_parity()}, "weights_checked": args.weights,
              "limits": "Torch DiT backend/TE compatibility shim is not a bit-equivalence proof to full upstream runtime. No environment result is implied."}
    if args.weights:
        model, vae, _, _ = native.load_model(device="cpu")
        result["checkpoint_sha256"] = native.MODEL_SHA256
        result["tokenizer_sha256"] = native.TOKENIZER_SHA256
        result["loaded_dit_tensors"] = model.checkpoint_adapter_metadata["loaded_tensors"]
        result["dit_parameters"] = sum(p.numel() for p in model.parameters())
        result["vae_parameters"] = sum(p.numel() for p in vae.parameters())
        del model
        video = torch.randn(1, 3, 33, 32, 32, generator=torch.Generator().manual_seed(14)).to(torch.bfloat16)
        z = vae.encode(video)
        assert z.shape == (1, 16, 9, 4, 4) and torch.isfinite(z).all()
        decoded = vae.decode(z)
        assert decoded.shape == video.shape and torch.isfinite(decoded).all()
        result["vae_cpu_encode_decode_shapes"] = [list(z.shape), list(decoded.shape)]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
