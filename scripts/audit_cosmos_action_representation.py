#!/usr/bin/env python3
"""CPU action-codec audit; imports no model/environment and never selects by outcome.

The reusable projection operates on the input tensor's device. CPU equality is
not evidence of CUDA equality: rerun this function on the deployment device.
Only a bounded, exact numerical gate is used; failure raises, never allclose.
"""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODECS = ROOT / "scripts/cosmos_policy_codecs.py"


@lru_cache(maxsize=1)
def official_codecs():
    """Load unmodified pure official functions without image/model imports."""
    tree = ast.parse(CODECS.read_text())
    names = {"extract_action_chunk_from_latent_sequence", "replace_latent_with_action_chunk"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != 2:
        raise RuntimeError("Official codec functions missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(CODECS), "exec"), namespace)
    return tuple(namespace[name] for name in (
        "extract_action_chunk_from_latent_sequence", "replace_latent_with_action_chunk"))


def array_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


class ExactProjectionError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("Action decoder exact projection failed: " + json.dumps(report))


@torch.inference_mode()
def project_action_slot(latent, mode="reverse_copies", *, max_correction_steps=32,
                        neighbor_steps=512):
    """Change representation, then correct final copy to exact original decode.

    No forecasts, rewards, or environment observations enter this function.
    Only slot 4 changes; correction touches only copy 111 of its 112 copies.
    Parameters are bounded in advance. Neighbor search accepts bit equality
    only, with deterministic plus-then-minus order and no best-outcome search.
    """
    if latent.dtype != torch.float32 or tuple(latent.shape[1:]) != (16, 9, 28, 28):
        raise ValueError("Expected native float32 latent (B,16,9,28,28)")
    if latent.ndim != 5 or latent.shape[0] != 1 or not torch.isfinite(latent).all():
        raise ValueError("Expected one finite native candidate")
    if max_correction_steps < 0 or neighbor_steps < 0:
        raise ValueError("Iteration bounds must be nonnegative")
    extract, replace = official_codecs()
    indices = torch.full((latent.shape[0],), 4, dtype=torch.int64, device=latent.device)
    target = extract(latent, (16, 7), indices).clone()
    target_flat = target.reshape(latent.shape[0], 112)
    changed = latent.clone()
    if mode == "reverse_copies":
        copies = latent[:, :, 4].reshape(latent.shape[0], 112, 112).flip(1).clone()
        changed[:, :, 4] = copies.reshape(latent.shape[0], 16, 28, 28)
    elif mode == "canonical_tile":
        changed = replace(changed, target, indices)
        copies = changed[:, :, 4].reshape(latent.shape[0], 112, 112).clone()
    else:
        raise ValueError("Unsupported fixed representation operation")
    before_projection = copies.clone()

    def decode():
        changed[:, :, 4] = copies.reshape(latent.shape[0], 16, 28, 28)
        return extract(changed, (16, 7), indices).reshape(latent.shape[0], 112)

    initial = decode().clone()
    correction_iterations = 0
    current = initial
    for step in range(max_correction_steps):
        unresolved = current != target_flat
        if not bool(unresolved.any()):
            break
        correction = 112.0 * (target_flat - current)
        copies[:, -1] = torch.where(unresolved, copies[:, -1] + correction, copies[:, -1])
        current = decode()
        correction_iterations = step + 1

    # Search a bounded symmetric ULP neighborhood of the last correction.
    # Different coordinates decode independently, so solved coordinates remain frozen.
    neighbor_iterations = 0
    plus, minus = copies[:, -1].clone(), copies[:, -1].clone()
    for step in range(neighbor_steps):
        if torch.equal(current, target_flat):
            break
        plus = torch.nextafter(plus, torch.full_like(plus, float("inf")))
        minus = torch.nextafter(minus, torch.full_like(minus, float("-inf")))
        for candidate in (plus, minus):
            unresolved = current != target_flat
            baseline = copies[:, -1].clone()
            copies[:, -1] = torch.where(unresolved, candidate, baseline)
            trial = decode()
            newly_solved = trial == target_flat
            copies[:, -1] = torch.where(unresolved & newly_solved, candidate, baseline)
            current = decode()
        neighbor_iterations = step + 1

    final = decode().reshape_as(target)
    other_equal = torch.equal(latent[:, :, :4], changed[:, :, :4]) and torch.equal(latent[:, :, 5:], changed[:, :, 5:])
    correction = copies - before_projection
    report = {
        "mode": mode, "device": str(latent.device), "dtype": str(latent.dtype),
        "shape": list(latent.shape), "decoder_target_source": "original raw latent decoded on same device",
        "max_correction_steps": max_correction_steps, "neighbor_steps": neighbor_steps,
        "correction_iterations": correction_iterations, "neighbor_iterations": neighbor_iterations,
        "unprojected_decoder_exact": torch.equal(initial, target_flat),
        "unprojected_decoder_max_abs_delta": float((initial - target_flat).abs().max()),
        "projected_decoder_exact": torch.equal(final, target),
        "projected_decoder_different_elements": int((final != target).sum()),
        "projected_decoder_max_abs_delta": float((final - target).abs().max()),
        "other_slots_bitwise_equal": other_equal,
        "correction_only_final_copy": bool((correction[:, :-1] == 0).all()),
        "correction_max_abs": float(correction.abs().max()),
        "correction_rms_over_final_copy": float(correction[:, -1].square().mean().sqrt()),
        "representation_changed": not torch.equal(latent, changed),
        "original_decoded_action_sha256": array_hash(target),
        "projected_decoded_action_sha256": array_hash(final),
        "original_action_slot_sha256": array_hash(latent[:, :, 4]),
        "projected_action_slot_sha256": array_hash(changed[:, :, 4]),
        "cuda_equivalence_proven": False,
    }
    if not report["projected_decoder_exact"] or not other_equal or not report["correction_only_final_copy"]:
        raise ExactProjectionError(report)
    return changed, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "runs/cosmos_counterfactual_ranking_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cosmos_action_representation_bus_audit_v1/cpu_exact_decoder_gate.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    extract, _ = official_codecs()
    rows = []
    paths = sorted(args.input_root.glob("prefix_*/candidates/candidate_*_latent.pt"))
    if not paths:
        raise RuntimeError("No saved native candidates")
    for path in paths:
        latent = torch.load(path, map_location="cpu", weights_only=True)
        target = extract(latent, (16, 7), torch.tensor([4]))
        copies = latent[:, :, 4].reshape(1, 112, 112)
        stored_normalized = torch.load(path.with_name(path.name.replace("_latent.pt", "_normalized.pt")), map_location="cpu", weights_only=True)
        row = {"path": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "shape": list(latent.shape), "dtype": str(latent.dtype),
               "action_copy_residual_rms": float((copies - target.reshape(1, 1, 112)).square().mean().sqrt()),
               "action_copy_residual_max_abs": float((copies - target.reshape(1, 1, 112)).abs().max()),
               "cpu_decode_equals_saved_native_normalized_action": torch.equal(target, stored_normalized),
               "cpu_decode_vs_saved_native_max_abs": float((target - stored_normalized).abs().max()),
               "projections": {}}
        for mode in ("reverse_copies", "canonical_tile"):
            try:
                _, report = project_action_slot(latent, mode)
                row["projections"][mode] = report
            except ExactProjectionError as error:
                row["projections"][mode] = error.report
        rows.append(row)
    result = {
        "kind": "CPU numerical codec gate; no model inference or environment experiment",
        "torch_version": torch.__version__, "threads": args.threads,
        "cuda_initialized": torch.cuda.is_initialized(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codec_source_sha256": hashlib.sha256(CODECS.read_bytes()).hexdigest(),
        "protocol": "All existing six preselected candidates; both fixed transformations; final-copy correction only; bounded numerical equality search independent of predictions/utility. No candidate excluded after failure.",
        "rows": rows,
        "passed_per_mode": {mode: sum(row["projections"][mode]["projected_decoder_exact"] for row in rows)
                            for mode in ("reverse_copies", "canonical_tile")},
        "candidate_count": len(rows),
        "limitations": ["CPU gate cannot certify CUDA reduction equivalence.",
                        "A future deployment-device gate must also compare decoded actions to the already committed native controls.",
                        "Exact action equality proves equal specified controls, not forecast error, ranking harm, or attacker access.",
                        "Projected canonical tile has a tiny correction in its final copy and is not an exact repeated tensor."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed_per_mode": result["passed_per_mode"],
                      "candidate_count": len(rows), "cuda_initialized": result["cuda_initialized"]}))


if __name__ == "__main__":
    main()
