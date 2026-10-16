#!/usr/bin/env python3
"""Decode six existing futures with the native placeholder-restoration path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosmos_cached_decode_gate_20261016.json"


def main():
    cfg = json.loads(CONFIG.read_text())
    source, out = ROOT / cfg["source"], ROOT / cfg["output"]
    if out.exists():
        raise FileExistsError("Preserve prior attempts; output already exists")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(CONFIG.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    counts = dict.fromkeys(cfg["budgets"], 0)
    report = {"status": "checking_inputs", "activity": counts, "scope": cfg["scope"],
              "metrics": [], "images": [], "decision_rule": cfg["decision_rule"]}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - start
        tmp = out / "summary.tmp.json"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "summary.json")

    def deadline(*_):
        raise TimeoutError("Frozen decode wall limit reached")

    persist()
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, cfg["resources"]["hard_wall_seconds"])
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        devices = [[int(x.strip()) for x in row.split(",")] for row in rows.strip().splitlines()]
        gpu = next(row for row in devices if row[0] == cfg["resources"]["gpu_index"])
        report["gpu_at_start"] = dict(zip(["index", "free_mib", "total_mib", "utilization_percent"], gpu))
        if gpu[1] < cfg["resources"]["minimum_free_mib"]:
            report["status"] = "resource_deferred_before_loading"
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu[0])
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import cosmos_policy_native_inference as native

        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(cfg["resources"]["memory_fraction"], 0)
        torch.cuda.reset_peak_memory_stats(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report["runtime"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "device": "cuda:0", "vae_parameter_dtype": "bfloat16"}
        assert all(report["runtime"][key] == value for key, value in cfg["required_runtime"].items())
        report["decode_argument_dtype"] = "float32 for every condition"
        prior = json.loads((source / "summary.json").read_text())
        assert prior["status"] == "completed_calibrated_first_query_truth_countercheck"
        assert cfg["candidates"] == [0, 1] and cfg["order"] == ["truth", "clean", "swap"]
        assert cfg["restored_placeholder_latent_indices"] == [0, 1, 4, 5]
        assert cfg["future_pixel_indices"] == {"wrist": 21, "primary": 25}

        def cached(name):
            return torch.load(source / name, map_location="cpu", weights_only=False)

        context = cached("context.pt")
        original = context["original_vae_latent"]
        assert tuple(original.shape) == tuple(cfg["latent_shape"])
        arrays, truths = {}, {}
        for i in cfg["candidates"]:
            raw = np.load(source / f"truth_candidate{i}_raw33.npy", allow_pickle=False)
            assert raw.shape == (33, 224, 224, 3) and raw.dtype == np.uint8
            truths[i] = cached(f"truth_candidate{i}_input.pt")
            # Joint encoding rather than the numerically injected action/proprio slots.
            arrays["truth", i] = truths[i]["joint_vae_latent"]
            assert torch.equal(arrays["truth", i][:, :, :5], original[:, :, :5])
            prepared_value = truths[i]["value_input"].clone()
            for index in cfg["restored_placeholder_latent_indices"]:
                prepared_value[:, :, index] = original[:, :, index]
            assert torch.equal(prepared_value[:, :, :8].to(torch.bfloat16),
                               arrays["truth", i][:, :, :8].to(torch.bfloat16))
            for condition in ("clean", "swap"):
                arrays[condition, i] = cached(f"{condition}_candidate{i}.pt")["future_latent"]
            for camera, frame in cfg["future_pixel_indices"].items():
                name = f"candidate{i}_{camera}_actual.png"
                Image.fromarray(raw[frame]).save(out / name)
                report["images"].append(name)
            for condition in cfg["order"]:
                x = arrays[condition, i]
                assert tuple(x.shape) == tuple(cfg["latent_shape"]) and torch.isfinite(x).all()
        report["status"] = "loading_vae"
        report["truth_value_decode_prefix_matches_joint_encoding_bf16"] = True
        report["frame_semantics"] = "33 modality-packed frames; future wrist/primary are the same control16 endpoint, not different physical times."
        persist()
        # Constructor strict-loads the same already-qualified local tokenizer.
        # No full policy checkpoint or network resource is loaded.
        counts["vae_loads"] += 1
        vae = native.OfficialWanVAE(device="cuda", verify=False)
        report["checkpoint"] = str(native.TOKENIZER.relative_to(ROOT))
        report["placeholder_restoration_equal_to_joint_truth"] = {}
        decoded = {}
        for condition in cfg["order"]:
            for i in cfg["candidates"]:
                report["status"] = "decoding"
                report["active_decode"] = {"condition": condition, "candidate": i}
                persist()
                x = arrays[condition, i]
                if condition == "truth":
                    restored = x.clone()
                    for index in cfg["restored_placeholder_latent_indices"]:
                        restored[:, :, index] = original[:, :, index]
                    same = torch.equal(restored, x)
                    report["placeholder_restoration_equal_to_joint_truth"][str(i)] = same
                    if not same:
                        raise ValueError("Native restoration would alter the joint truth encoding")
                counts["vae_decodes"] += 1
                assert counts["vae_decodes"] <= cfg["budgets"]["vae_decodes"]
                # The helper scales and quantizes in its argument dtype. Match the
                # FP32 native Future/Value inputs for every condition, including truth.
                images = native.decode_future_images(vae, x.to(device="cuda", dtype=torch.float32),
                                                     original.to(device="cuda", dtype=torch.float32))
                torch.cuda.synchronize()
                raw = np.load(source / f"truth_candidate{i}_raw33.npy", allow_pickle=False)
                for camera, frame in cfg["future_pixel_indices"].items():
                    rgb = images[camera][0]
                    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
                    decoded[condition, i, camera] = rgb
                    name = f"candidate{i}_{camera}_{condition}.png"
                    Image.fromarray(rgb).save(out / name)
                    report["images"].append(name)
                    error = rgb.astype(np.float64) - raw[frame].astype(np.float64)
                    mse = float(np.mean(error ** 2))
                    report["metrics"].append({"condition": condition, "candidate": i,
                        "camera": camera, "pixel_mae_0_255": float(np.mean(np.abs(error))),
                        "pixel_mse_0_255": mse,
                        "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
                        "interpretation": "RGB discrepancy, not physical-state error or utility"})
                persist()
        assert counts == cfg["budgets"]
        # A scientific contact sheet, not an inferred state or automatic semantic judgment.
        sheet = Image.new("RGB", (4 * 224 + 150, 4 * 252 + 36), "white")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(("Measured RGB", "Truth VAE", "Clean forecast", "Swap forecast")):
            draw.text((150 + 224 * col + 8, 10), label, fill="black")
        for row, (i, camera) in enumerate((i, c) for i in (0, 1) for c in ("wrist", "primary")):
            draw.text((5, 36 + row * 252 + 88), f"Candidate {i}\n{camera}", fill="black")
            for col, condition in enumerate(("actual", "truth", "clean", "swap")):
                img = Image.open(out / f"candidate{i}_{camera}_{condition}.png")
                sheet.paste(img, (150 + 224 * col, 36 + row * 252))
        sheet.save(out / "contact_sheet.png")
        report["contact_sheet"] = str((out / "contact_sheet.png").relative_to(ROOT))
        report["status"] = "completed_six_cached_native_decodes"
        report["active_decode"] = None
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        report["limits"] = ["No physical state estimator was calibrated or executed.",
            "Endpoint drawer/bowl/contact state is absent from the existing replay cache.",
            "Decode reconstruction and forecast image differences do not isolate content from style.",
            "No new ranking, actual task outcome, defense or independent generalization was measured."]
    except Exception as exc:
        report["status"] = "failed_or_resource_limited"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        persist()
    print(json.dumps({"status": report["status"], "activity": counts,
                      "seconds": report["elapsed_seconds"], "output": str(out.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
