#!/usr/bin/env python3
"""Verify saved cached-metadata experiment artifacts on CPU; no model or env.

Rehashes tensor contents rather than trusting summary digests. A saved adapter
cannot independently prove unsaved post-training base weights: that check is
explicitly reported as a runtime-recorded assertion only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = value.detach().contiguous().cpu()
        h.update(key.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_tensors(path):
    values = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(values, dict)
    assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in values.items())
    assert all(v.device.type == "cpu" and bool(torch.isfinite(v).all()) for v in values.values())
    return values


def verify(path, checkpoint_sha):
    doc = json.loads(path.read_text())
    assert doc["status"] == "completed"
    protocol = doc["protocol"]
    assert protocol["checkpoint_sha256"] == checkpoint_sha
    checked_sources = {}
    for relative, expected in protocol["source_files"].items():
        actual = file_sha(ROOT / relative)
        assert actual == expected, f"Changed experiment source: {relative}"
        checked_sources[relative] = actual
    source = protocol["source"]
    for name in ("summary", "trace"):
        assert file_sha(source[f"{name}_path"]) == source[f"{name}_sha256"]
    assert file_sha(doc["data"]["path"]) == doc["data"]["sha256"]
    init_path = path.parent / "initial_adapter.pt"
    initial = load_tensors(init_path)
    digest = tensor_sha(initial)
    assert digest == doc["initial_adapter_sha256"]
    assert len(initial) == 2 * doc["lora_projection_count"]
    assert sum(x.numel() for x in initial.values()) == doc["trainable_parameter_count"]
    assert all(bool((v == 0).all()) for k, v in initial.items() if k.endswith(".lora_B"))
    arms = {}
    for condition, row in doc["results"].items():
        adapter_path, grad_path = path.parent / row["adapter_file"], path.parent / row["gradient_file"]
        adapter, gradients = load_tensors(adapter_path), load_tensors(grad_path)
        assert adapter.keys() == initial.keys()
        assert tensor_sha(adapter) == row["adapter_sha256"]
        changed = [k for k in initial if not torch.equal(initial[k], adapter[k])]
        grad_a_zero = all(bool((v == 0).all()) for k, v in gradients.items() if k.endswith(".lora_A"))
        grad_b_nonzero = sum(bool((v != 0).any()) for k, v in gradients.items() if k.endswith(".lora_B"))
        grad_norm = math.sqrt(sum(float(v.double().square().sum()) for v in gradients.values()))
        if condition == "Frozen":
            assert not changed and not gradients
            source_ep = next(e for e in row["episodes"] if e["init_id"] == protocol["source_init_id"])
            assert file_sha(source_ep["trace"]) == source["trace_sha256"]
        else:
            assert gradients.keys() == initial.keys()
            assert tensor_sha(gradients) == row["training"]["gradient_sha256"]
            assert changed and grad_a_zero and grad_b_nonzero > 0
            assert 0 < grad_norm <= .100001
        arms[condition] = {
            "adapter_file_sha256": file_sha(adapter_path), "adapter_tensor_sha256": tensor_sha(adapter),
            "gradient_file_sha256": file_sha(grad_path), "gradient_tensor_sha256": tensor_sha(gradients),
            "tensor_count": len(adapter), "all_tensors_finite": True,
            "changed_parameter_tensor_count": len(changed), "gradient_l2": grad_norm,
            "zero_B_initialization_A_gradients_zero": grad_a_zero,
            "nonzero_B_gradient_tensor_count": grad_b_nonzero,
            "frozen_source_trace_file_exact": True if condition == "Frozen" else None,
        }
    metadata = {}
    # Only trusted files generated locally by these frozen runners are loaded.
    for name, item in doc["cached_metadata"].items():
        assert file_sha(item["file"]) == item["sha256"]
        value = torch.from_numpy(np.load(item["file"], allow_pickle=True).item()["text"])
        actual = tensor_sha({"text": value})
        assert actual == item["token_sha256"] and list(value.shape) == item["shape"]
        metadata[name] = {"file_sha256": item["sha256"], "tensor_sha256": actual,
                          "literal_recorded": item["literal_text"], "shape": list(value.shape)}
    before, after = doc["frozen_parameter_sha256_before"], doc["frozen_parameter_sha256_after"]
    assert before == after and doc["frozen_parameters_unchanged"] is True
    return {
        "summary": str(path), "summary_sha256": file_sha(path),
        "protocol_identity_sha256": doc["protocol_identity_sha256"],
        "source_files_rehashed": checked_sources,
        "physical_cache_sha256": doc["data"]["sha256"],
        "initial_adapter_file_sha256": file_sha(init_path), "initial_adapter_tensor_sha256": digest,
        "all_initial_B_tensors_zero": True, "arms": arms, "cached_metadata": metadata,
        "base_parameter_integrity": {
            "runtime_recorded_before_after_equal": True, "recorded_tensor_sha256": before,
            "independent_post_training_base_tensor_check": False,
            "limitation": "No post-training frozen-base checkpoint was saved. CPU audit verifies original checkpoint and immutable sources, but cannot independently rehash the former live model's base parameters.",
        },
        "metadata_limitations": ["Token tensor/file digests independently checked; literal-to-token encoding is the runner's recorded contract.",
                                 "protocol.cpu_scope contains static CPU guidance even for CUDA runs; actual protocol.model_device and per-episode model_device identify the CUDA experiment."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--active-weight-audit", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    checkpoint = ROOT / "models/WorldAgen/libero-38.pth"
    checkpoint_sha = file_sha(checkpoint)
    records = [verify(p.resolve(), checkpoint_sha) for p in args.summaries]
    assert len({r["initial_adapter_tensor_sha256"] for r in records}) == 1
    whole_base_equal = len({r["base_parameter_integrity"]["recorded_tensor_sha256"] for r in records}) == 1
    active_audit = None
    if not whole_base_equal:
        assert args.active_weight_audit is not None, "Different recorded frozen-base digests require an explicit source/active-weight audit"
        proof = json.loads(args.active_weight_audit.read_text())
        assert proof["status"] == "passed_cpu_cached_window_intervention"
        assert proof["source_derived_active_weight_equivalence"] is True
        assert proof["policy_world_loss_and_gradient_bitwise_equal"] is True
        assert proof["cache_file_sha256"] == records[0]["physical_cache_sha256"]
        assert all(file_sha(p) == h for p, h in proof["source_files"].items())
        constructor = Path(proof["constructor_audit_file"])
        assert file_sha(constructor) == proof["constructor_audit_sha256"]
        constructor_doc = json.loads(constructor.read_text())
        assert constructor_doc["all_other_parameters_exact"] is True
        assert all(file_sha(p) == h for p, h in constructor_doc["source_paths"].items())
        active_audit = {"path": str(args.active_weight_audit.resolve()),
                        "sha256": file_sha(args.active_weight_audit),
                        "status": "source_derived_active_weight_equivalence",
                        "direct_historical_active_parameter_digest_equality": False}
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "cpu_only": True, "cuda_initialized": False,
              "verifier_sha256": file_sha(__file__), "checkpoint_file_sha256": checkpoint_sha,
              "staged_initial_adapter_tensors_identical": True,
              "staged_recorded_base_parameter_digests_identical": whole_base_equal,
              "active_weight_audit": active_audit, "runs": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "out": str(args.out), "cuda_initialized": False,
                      "runs": [{"summary": r["summary"], "arms": list(r["arms"])} for r in records]}))


if __name__ == "__main__":
    main()
