#!/usr/bin/env python3
"""Real-snapshot CUDA parity/resource check; no DiT, environment, or inference edits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

import cosmos_policy_native_inference as native
from cosmos_policy_cuda_branch_offload import BranchOffloadedVAE, branch_audit
from cosmos_policy_cuda_offload import OffloadedVAE
from pilot_cosmos_counterfactual_action_ranking import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(json.dumps({"shape": list(t.shape), "dtype": str(t.dtype)}, sort_keys=True).encode())
    h.update(b"\0")
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_status():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
                                    "--format=csv,noheader"], text=True).strip()


class Capture:
    def __init__(self, vae):
        self.vae, self.inputs = vae, []

    @property
    def latents_mean(self):
        return self.vae.latents_mean

    def encode(self, value):
        self.inputs.append({"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)})
        return self.vae.encode(value)


def cpu_audit():
    with torch.device("meta"):
        core = native.WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                              attn_scales=[], temperal_downsample=[False, True, True],
                              dropout=0.0, temporal_window=16).to(torch.bfloat16)
    wrapper = torch.nn.Module()
    wrapper.vae = core
    for name in ("latents_mean", "latents_std", "inv_std"):
        wrapper.register_buffer(name, torch.empty(16, device="meta", dtype=torch.bfloat16))
    files = ["scripts/cosmos_policy_cuda_branch_offload.py", "scripts/check_cosmos_vae_branch_offload.py",
             "scripts/cosmos_policy_cuda_offload.py", "scripts/cosmos_policy_native_inference.py",
             "scripts/cosmos_policy_wan_vae.py", "scripts/cosmos_policy_codecs.py"]
    return {"status": "cpu_parameter_audit_complete", "parameter_audit": branch_audit(wrapper),
            "source_sha256": {f: native.digest(ROOT / f) for f in files},
            "torch_version": torch.__version__, "gpu_used": False,
            "tokenizer_expected_sha256": native.TOKENIZER_SHA256}


@torch.inference_mode()
def check(args, audit):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshots = []
    for prefix in (0, 16, 80):
        obs, contract = load_snapshot(ROOT / f"runs/cosmos_counterfactual_snapshots_v1/prefix_{prefix:03d}/snapshot.json")
        snapshots.append((obs, contract))
    result = {"status": "running", "audit": audit, "snapshot_contracts": [c for _, c in snapshots],
              "gpu_before": gpu_status(), "rows": [], "scope": "VAE-only exact same-input CUDA parity; no full DiT/task guarantee",
              "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False}
    report = args.out_dir / "parity.json"
    write(report, result)
    base = native.OfficialWanVAE(device="cpu")
    if branch_audit(base) != audit["parameter_audit"]:
        raise RuntimeError("Loaded VAE differs from CPU parameter audit")
    stats = native.load_stats()
    outputs = {}

    def run_one(mode, method, index):
        prefix = snapshots[index][1]["prefix_steps"]
        row = {"mode": mode, "method": method, "prefix_steps": prefix, "status": "running",
               "gpu_before": gpu_status()}
        result["rows"].append(row)
        write(report, result)
        wrapper = OffloadedVAE(base, "cuda") if mode == "full_reference" else BranchOffloadedVAE(base)
        capture = Capture(wrapper)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        row["cuda_baseline_allocated_bytes"] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        values = None
        try:
            if method == "encode":
                values = native.encode_observation(capture, snapshots[index][0], stats)
                row["inputs"] = capture.inputs
            else:
                original = outputs[("full_reference", "encode", index)][1].to("cuda")
                row["inputs"] = [{"shape": list(original.shape), "dtype": str(original.dtype),
                                  "sha256": tensor_hash(original)}]
                values = (wrapper.decode(original),)
                del original
            torch.cuda.synchronize()
            outputs[(mode, method, index)] = tuple(v.detach().cpu() for v in values)
            row["outputs"] = [{"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_hash(v)} for v in values]
            row["status"] = "completed"
            if mode == "branch":
                ref_key = ("full_reference", method, index)
                if ref_key in outputs:
                    ref_row = next(r for r in result["rows"] if r["mode"] == "full_reference"
                                   and r["method"] == method and r["prefix_steps"] == prefix)
                    row["input_bitwise_equal"] = row["inputs"] == ref_row["inputs"]
                    row["output_bitwise_equal"] = all(torch.equal(a, b) for a, b in zip(outputs[ref_key], outputs[(mode, method, index)]))
                    row["max_abs_difference"] = max(float((a.float() - b.float()).abs().max()) for a, b in zip(outputs[ref_key], outputs[(mode, method, index)]))
                    if not row["input_bitwise_equal"] or not row["output_bitwise_equal"]:
                        row["status"] = "parity_failed"
                else:
                    row["parity_unavailable_reason"] = "Full-residency reference did not complete"
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "cuda_out_of_memory"
            row["error"] = str(exc)
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            base.vae.clear_cache()
            base.vae.to("cpu")
            row["all_core_parameters_cpu_after_call"] = all(p.device.type == "cpu" for p in base.vae.parameters())
            del values, capture, wrapper
            gc.collect()
            torch.cuda.empty_cache()
            row["gpu_after"] = gpu_status()
            write(report, result)
            print(json.dumps(row), flush=True)
        return row["status"] == "completed"

    for mode in ("full_reference", "branch"):
        for index in range(3):
            if not run_one(mode, "encode", index):
                break  # Never retry the same allocation failure on more snapshots.
    if args.check_decode and all(("full_reference", "encode", i) in outputs for i in range(3)):
        for mode in ("full_reference", "branch"):
            for index in range(3):
                if not run_one(mode, "decode", index):
                    break
    else:
        result["decode_status"] = "not_requested_or_reference_encode_unavailable"
    checks = [r for r in result["rows"] if r["mode"] == "branch" and r["method"] == "encode"]
    result["encode_all_three_bitwise_equal"] = len(checks) == 3 and all(r.get("output_bitwise_equal") and r.get("input_bitwise_equal") for r in checks)
    dec = [r for r in result["rows"] if r["mode"] == "branch" and r["method"] == "decode"]
    result["decode_all_three_bitwise_equal"] = len(dec) == 3 and all(r.get("output_bitwise_equal") and r.get("input_bitwise_equal") for r in dec)
    result["status"] = "encode_parity_passed" if result["encode_all_three_bitwise_equal"] else "encode_parity_not_established"
    if result["decode_all_three_bitwise_equal"]:
        result["status"] = "encode_and_decode_parity_passed"
    result["gpu_after"] = gpu_status()
    write(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/cosmos_vae_branch_offload_parity_v1")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--check-decode", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = cpu_audit()
    audit_path = args.out_dir / "cpu_audit.json"
    if args.cpu_only:
        if audit_path.exists():
            raise FileExistsError(audit_path)
        write(audit_path, audit)
        print(json.dumps(audit, indent=2))
        return
    if not audit_path.exists() or json.loads(audit_path.read_text()) != audit:
        raise RuntimeError("Run the matching --cpu-only audit before CUDA")
    if (args.out_dir / "parity.json").exists():
        raise FileExistsError("Keep prior parity evidence; use a fresh output directory")
    result = check(args, audit)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "snapshot_contracts", "audit"}}, indent=2))


if __name__ == "__main__":
    main()
