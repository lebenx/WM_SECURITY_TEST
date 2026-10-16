#!/usr/bin/env python3
"""Read-only CPU diagnosis of the fixed Task8/Init1 image parity failure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "source_n2": ("runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/libero_goal_task08_init01_clean_initial.png", "ca12a8fdbea175841d3ea2608f613d75737d2bada944721828b794cefa980042"),
    "actual": ("runs/cosmos_first_candidate_utility_matched30_v1/task08_init01/candidate1/libero_goal_task08_init01_clean_initial.png", "7f9e6e05ce1f396d1c566b911528676e0da9727691f2e0dd78c6f90f7fc4d1f5"),
    "old_n1": ("runs/cosmos_policy_native_task_gate_clean_n1_matched30_v1/libero_goal_task08_init01_clean_initial.png", "ca12a8fdbea175841d3ea2608f613d75737d2bada944721828b794cefa980042"),
}
CODEC_SHA = "6cf536f1932a105d5cc033ed6ea5bc82327572b1e495510a32f2c690181dc330"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_record(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return {"dtype": a.dtype.str, "shape": list(a.shape), "bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "array_sha256": hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()}


def compare(left, right, include_coordinates=False):
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("Mismatched image format")
    d = b.astype(np.int16) - a.astype(np.int16)
    nz = np.any(d != 0, axis=-1)
    ys, xs = np.nonzero(nz)
    values, counts = np.unique(d[d != 0], return_counts=True)
    result = {"exact": a.tobytes() == b.tobytes(), "source": array_record(a), "compared": array_record(b),
              "different_pixels": int(nz.sum()), "different_channel_values": int(np.count_nonzero(d)),
              "max_abs": int(np.abs(d).max()), "mse_0_255": float(np.mean(d.astype(np.float64) ** 2)),
              "mse_0_1": float(np.mean((d.astype(np.float64) / 255.) ** 2)),
              "bounding_box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
              "signed_channel_difference_counts": {str(v): int(n) for v, n in zip(values, counts)}}
    if include_coordinates:
        result["different_pixel_coordinates_yx"] = np.argwhere(nz).tolist()
        result["different_channel_coordinates_yxc"] = np.argwhere(d != 0).tolist()
    return result


def analyze():
    codec_path = ROOT / "scripts/cosmos_policy_codecs.py"
    if sha(codec_path) != CODEC_SHA:
        raise ValueError("Frozen codec source changed")
    import cosmos_policy_codecs as codecs
    import torch
    import torchvision
    import PIL
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic must not initialize CUDA")
    inputs, stages = {}, {}
    for name, (relative, expected) in INPUTS.items():
        path = ROOT / relative
        digest = sha(path)
        if digest != expected:
            raise ValueError(f"Fixed diagnostic input changed: {relative}")
        with Image.open(path) as image:
            array = np.asarray(image).copy()
            if image.mode != "RGB" or array.shape != (256, 512, 3):
                raise ValueError("Unexpected combined native PNG format")
            inputs[name] = {"path": str(path), "sha256": digest, "file_bytes": path.stat().st_size,
                            "mode": image.mode, "size": list(image.size), "metadata": image.info,
                            "decoded_array": array_record(array)}
        # native.frame already flips raw observations vertically and joins primary,wrist.
        views = np.stack([array[:, :256], array[:, 256:]], axis=0)
        jpeg = codecs.apply_jpeg_compression_np(views, quality=95)
        resized = codecs.resize_images(jpeg, 224)
        final = np.asarray(codecs.prepare_images_for_model(list(views), SimpleNamespace(
            use_jpeg_compression=True, trained_with_image_aug=True)))
        stages[name] = {"raw_display_orientation": views, "jpeg95": jpeg,
                        "jpeg95_resize224": resized, "official_complete224": final}
    comparisons = {}
    for other in ("actual", "old_n1"):
        comparisons["source_n2_vs_" + other] = {
            stage: {view: compare(stages["source_n2"][stage][i], stages[other][stage][i],
                                  include_coordinates=stage == "raw_display_orientation")
                    for i, view in enumerate(("primary", "wrist"))}
            for stage in stages["source_n2"]}
    return {"schema_version": 1, "status": "completed_cpu_diagnostic", "inputs": inputs,
            "code": {str(p.relative_to(ROOT)): sha(p) for p in (
                Path(__file__).resolve(), codec_path, ROOT / "scripts/cosmos_policy_native_inference.py",
                ROOT / "scripts/pilot_cosmos_policy_native_task_gate.py")},
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__, "Pillow": PIL.__version__},
            "comparisons": comparisons,
            "orientation": "Stored native PNG: primary then wrist, each already vertically flipped. No second flip; each view processed independently by unchanged official CPU codec.",
            "conclusion_scope": "Image/container and deterministic input-codec diagnosis only. Raw wrist pixels and official model-input pixels differ; primary and old N1 reference are exact. No inference about hidden physical/renderer causes, model output or task utility.",
            "gpu_used": False, "cuda_initialized": torch.cuda.is_initialized(), "environment_created": False,
            "checkpoint_loaded": False, "model_forward_calls": 0, "training_updates": 0, "api_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/cosmos_initial_observation_parity_cpu_v1/analysis.json")
    args = parser.parse_args()
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.out.resolve()), "sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
