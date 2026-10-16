#!/usr/bin/env python3
"""One heldout-task rollout timing probe under recorded co-resident GPU load.

This does not change or restart the active exclusive-GPU Clean runner. No Clean
episode is executed, and no co-resident process is stopped or modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    sys.path.insert(0, str(ROOT / ".deps/nanowm_runtime"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                      VAE_MODEL_PATH=str(ROOT / "models/nanowm_sd_vae"))
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cuda")
    import pilot_nanowm_heldout_clean as heldout
    import pilot_nanowm_pointmaze_clean as base
    if out.exists():
        parser.error("Preserve prior attempts; output path already exists")
    out.mkdir(parents=True)
    summary = {"status": "started", "model_rollout_calls": 0, "clean_task_episodes": 0,
               "environment_steps": 0, "other_processes_modified": False,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "purpose": "Resolve current shared-capacity uncertainty after a change from three forecast jobs to two smaller-memory training jobs; one probe only, no best-of timing search"}
    env = None
    base.write_json(out / "summary.json", summary)
    try:
        protocol = json.loads(heldout.PROTOCOL.read_text())
        cfg, native, manifest, cache = heldout.prepare_config(args, out, protocol)
        rows, images = heldout.load_targets(protocol)
        manifest.update(probe_runner=base.identity(__file__), input_images=images,
                        probe_scope="Timing only under observed co-resident load; not an execution of the exclusive-GPU Clean protocol")
        base.write_json(out / "runtime_manifest.json", manifest)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before the capacity check")
        before = heldout.gpu_sample(0)
        summary["gpu_before_load"] = before
        free_mib = int(before["memory_total_mib"]) - int(before["memory_used_mib"])
        if free_mib < 6144:
            summary.update(status="resource_deferred", reason="Less than the fixed 6 GiB free-memory threshold")
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = before["uuid"]
        torch.set_float32_matmul_precision("high")
        base.seed_all(42)
        from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler
        isolate_d4rl_offline_env()
        prepare_legacy_compiler()
        env = native.create_planning_env("point_maze", device="cpu")
        experiment, _, wm, _ = base.build_runtime(native, cfg, cache, torch.device("cuda"))
        summary["strict_checkpoint_load_passed"] = True

        def record_capacity(stage):
            sample = heldout.gpu_sample(0)
            if sample["uuid"] != before["uuid"]:
                raise RuntimeError("Physical GPU changed")
            sample["stage"] = stage
            base.append_json(out / "gpu_samples.jsonl", sample)

        summary["cost_probe"] = heldout.cost_probe(experiment, wm, env, rows[0], out / "cost_probe",
                                                    torch.device("cuda"), record_capacity)
        summary["model_rollout_calls"] = 1
        estimate = summary["cost_probe"]["estimated_two_plan_rollout_seconds"]
        summary.update(status="completed_shared_capacity_probe", within_2700_second_estimate=estimate <= 2700,
                       limit="A one-shot cold wall-time estimate under this observed load, not intrinsic throughput or a guarantee of future runtime. No automatic Clean launch or protocol change follows.",
                       gpu_after=heldout.gpu_sample(0))
    except BaseException as error:
        summary.update(status="technical_error", error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        base.write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
