#!/usr/bin/env python3
"""Measure native rollout components on an existing frozen observation.

No planner, environment, precision change, or sampler replacement is used.
CUDA events delimit native VAE encoding and the 40 native model forwards.
Co-resident work can affect both device intervals and end-to-end latency.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        parser.error("Preserve prior results; output directory already exists")
    sys.path.insert(0, str(ROOT / ".deps/nanowm_runtime"))
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cuda")
    args.checkpoint = ROOT / "models/nanowm_pointmaze/model.safetensors"
    args.vae_dir = ROOT / "models/nanowm_sd_vae"
    args.resource_summary = ROOT / "runs/nanowm_resource_gate_v2/summary.json"
    args.protocol = ROOT / "reports/nanowm_clean_gate_protocol_20261015.json"
    args.rollout_batch_size = 16
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                      VAE_MODEL_PATH=str(args.vae_dir))
    import numpy as np
    import torch
    from PIL import Image
    from pilot_nanowm_pointmaze_clean import preflight, build_runtime, seed_all, write_json

    args.out_dir.mkdir(parents=True)
    summary = {"status": "started", "task_episodes": 0, "environment_steps": 0,
               "rollout_calls": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "scope": "Instrumented resource profile, not task capability or security evidence",
               "gpu_before": gpu_snapshot()}
    write_json(args.out_dir / "summary.json", summary)
    try:
        cfg, native, manifest, cache = preflight(args, args.out_dir)
        write_json(args.out_dir / "runtime_manifest.json", manifest)
        torch.set_float32_matmul_precision("high")
        seed_all(42)
        experiment, _, wm, _ = build_runtime(native, cfg, cache, torch.device("cuda"))
        source = ROOT / "runs/nanowm_pointmaze_timing_v1"
        obs = {"visual": np.asarray(Image.open(source / "seed_42/initial.png").convert("RGB"))}
        visual = experiment._obs_to_tensor(obs, 256, torch.device("cuda")).expand(16, -1, -1, -1, -1)
        actions = torch.load(source / "benchmark_normalized_actions.pt", weights_only=True).cuda()
        # Reproduce the RNG point after the original CPU action proposal.
        seed_all(42)
        original_actions = torch.randn(16, 5, 10)
        original_actions[0] = 0
        if not torch.equal(actions.cpu(), original_actions):
            raise RuntimeError("Saved candidates differ from the original proposal")
        events = {"vae_encode": [], "model_forward": []}

        def timed(function, label):
            def wrapped(*fargs, **fkwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = function(*fargs, **fkwargs)
                end.record()
                events[label].append((start, end))
                return result
            return wrapped

        wm.latent_codec.encode = timed(wm.latent_codec.encode, "vae_encode")
        wm.model.forward = timed(wm.model.forward, "model_forward")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with torch.no_grad():
            result, _ = wm.rollout({"visual": visual}, actions)
        torch.cuda.synchronize()
        wall = time.perf_counter() - begin
        summary["rollout_calls"] = 1
        finite = bool(torch.isfinite(result["visual"]).all())
        if not finite or list(result["visual"].shape) != [16, 6, 4096]:
            raise RuntimeError("Invalid rollout output")
        components = {}
        for label, pairs in events.items():
            elapsed = [start.elapsed_time(end) / 1000 for start, end in pairs]
            components[label] = {"calls": len(elapsed), "device_interval_seconds": sum(elapsed),
                                 "per_call_seconds": elapsed}
        if components["model_forward"]["calls"] != 40 or components["vae_encode"]["calls"] != 1:
            raise RuntimeError("Unexpected native call count")
        vae_fraction = components["vae_encode"]["device_interval_seconds"] / wall
        summary.update(status="completed_native_component_profile", elapsed_seconds=wall,
                       components=components, output_shape=list(result["visual"].shape), finite=finite,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       vae_device_interval_over_wall=vae_fraction,
                       idealized_speedup_if_vae_interval_removed=1 / (1 - vae_fraction) if vae_fraction < 1 else None,
                       estimated_two_plan_rollout_seconds=wall * 7 * 30 * 2,
                       estimate_limits="Cold instrumented microbatch extrapolation with co-resident GPU work; excludes non-rollout overhead and is neither observed task time nor an isolated-hardware bound",
                       gpu_after=gpu_snapshot())
    except BaseException as error:
        summary.update(status="technical_error", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(args.out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
