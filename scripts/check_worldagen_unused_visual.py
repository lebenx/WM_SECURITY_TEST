#!/usr/bin/env python3
"""CPU-only audit of unused CLIP visual storage; not a task-success experiment.

Runs real cached WorldAgen inputs before/after the storage helper, compares all
outputs at a common RNG state, and backpropagates the native image MSE through
zero-B LoRA. No optimizer update, simulator, API, or CUDA allocation is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from worldagen_unused_visual import offload_unused_worldagen_branches


def compare(left, right):
    result = []
    for index, (a, b) in enumerate(zip(left, right)):
        if a is None or b is None:
            result.append({"output": index, "both_none": a is None and b is None,
                           "exact": a is None and b is None})
        else:
            result.append({"output": index, "shape": list(a.shape),
                           "exact": torch.equal(a, b), "max_abs": float((a - b).abs().max())})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=Path, default=Path("runs/worldagen_prediction_contract_cpu_v2/chunk_000"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    import pilot_worldagen_native_task_gate as gate
    import pilot_worldagen_forecast_ttt_task as ttt
    gate.base.DEVICE = "cpu"
    model = gate.base.build_model()
    counters = {"visual_forward_calls": 0, "text_forward_calls": 0}
    visual_hook = model.clip_model.visual.register_forward_pre_hook(
        lambda _module, _inputs: counters.__setitem__("visual_forward_calls", counters["visual_forward_calls"] + 1))
    text_hook = model.clip_model.token_embedding.register_forward_pre_hook(
        lambda _module, _inputs: counters.__setitem__("text_forward_calls", counters["text_forward_calls"] + 1))
    cached = torch.load(args.chunk_dir / "model_inputs.pt", map_location="cpu", weights_only=True)
    goals = json.loads((args.chunk_dir / "prediction_commitment.json").read_text())["goal_texts"]
    goal = goals[0] if isinstance(goals, list) else goals["0"]
    text = gate.base.clip.tokenize([goal]).unsqueeze(1).repeat(1, 2, 1)
    inputs = (cached["primary"][:1], cached["wrist"][:1], cached["state"][:1], text, cached["action"][:1])
    def run():
        torch.random.default_generator.manual_seed(20261014)
        return model(*inputs)
    with torch.inference_mode():
        before = run()
    storage = offload_unused_worldagen_branches(model)
    with torch.inference_mode():
        after = run()
    frozen_equivalence = compare(before, after)
    del before
    torch.random.default_generator.manual_seed(20261014)
    adapted = ttt.attach_lora(model)
    with torch.inference_mode():
        zero_lora = run()
    zero_lora_equivalence = compare(after, zero_lora)
    del after, zero_lora
    model.train()
    outputs = run()
    target = torch.load(args.chunk_dir / "targets_native_patchify.pt", map_location="cpu", weights_only=True)[:1]
    world = outputs[2].reshape(1, 2, 2, 196, 768)[:, 0]
    loss = .1 * ((world - target) ** 2).mean()
    loss.backward()
    gradients = ttt.adapter_parameters(model)
    grads_present = sum(parameter.grad is not None for parameter in gradients.values())
    grads_nonzero = sum(parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
                        for parameter in gradients.values())
    assert all(row["exact"] for row in frozen_equivalence + zero_lora_equivalence)
    assert counters["visual_forward_calls"] == 0
    assert counters["text_forward_calls"] == 4
    assert grads_nonzero > 0
    assert not torch.cuda.is_initialized()
    visual_hook.remove()
    text_hook.remove()
    try:
        model.clip_model.visual(torch.empty(0))
    except RuntimeError as error:
        guard_works = "CPU-offloaded CLIP visual was invoked" in str(error)
    else:
        guard_works = False
    assert guard_works
    try:
        model.vision_encoder.decoder_embed(torch.empty(0))
    except RuntimeError as error:
        decoder_guard_works = "CPU-offloaded MAE decoder was invoked" in str(error)
    else:
        decoder_guard_works = False
    assert decoder_guard_works
    report = {"status": "passed_cpu_static_storage_and_forward_audit", "storage": storage,
              "chunk_dir": str(args.chunk_dir.resolve()), "threads": args.threads,
              "native_outputs_before_after_offload": frozen_equivalence,
              "zero_lora_outputs": zero_lora_equivalence, **counters,
              "lora_projection_count": len(adapted), "lora_tensors_with_gradients": grads_present,
              "lora_tensors_with_nonzero_gradients": grads_nonzero,
              "forecast_loss_scaled_by_point_one": float(loss.detach()),
              "accidental_visual_call_guard_works": guard_works,
              "accidental_mae_decoder_call_guard_works": decoder_guard_works,
              "cuda_initialized": torch.cuda.is_initialized(), "optimizer_updates": 0,
              "limits": "CPU audit confirms unused execution path, dtype/key/storage preservation and gradients; actual CUDA memory transfer and task behavior require separate integration validation.",
              "elapsed_seconds": time.monotonic() - started}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
