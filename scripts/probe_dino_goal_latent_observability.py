#!/usr/bin/env python3
"""CPU native observation-encoder check; no prediction, planner or environment."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/dino_goal_observability_20261015"
OUTPUT = ROOT / "runs/dino_goal_latent_observability_20261015"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = dict(observation_encode=0, visual_encoder=0, proprio_encoder=0,
                  predictor=0, unroll=0, forward_pred=0, rollout=0, goal_encode=0,
                  environment_steps=0, planner_calls=0, network_attempts=0)
    report = {"status": "running_cpu_native_encoder_probe", "interpreter": sys.executable,
              "source": str(SOURCE.relative_to(ROOT)), "wall_time_limit_seconds": 180,
              "counts": counts, "checkpoint_load_calls": {},
              "scope": "One fixed physical state, two preselected visible goal markers and original/restored controls; actual native z_init only"}

    def persist():
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def stop(signum, frame):
        raise TimeoutError("CPU encoder probe exceeded its fixed wall-time limit")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, 180)
    persist()
    try:
        source = json.loads((SOURCE / "summary.json").read_text())
        if source["status"] != "completed_cpu_paired_goal_render_gate":
            raise ValueError("Wait for the completed render record; partial outputs are not inputs")
        if not all(source["restoration"].values()) or not all(
                all(row["physical_exact_to_baseline_before_and_after_render"].values())
                for row in source["renders"].values()):
            raise ValueError("Source physical/restoration controls did not pass")

        import numpy as np
        import torch

        torch.set_num_threads(4)
        report.update(torch_version=torch.__version__, torch_path=torch.__file__, numpy_version=np.__version__)
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
        labels = ("original", "goal_a", "goal_b", "restored")
        with np.load(SOURCE / "paired_observations.npz", allow_pickle=False) as arrays:
            pixels = {name: arrays[name + "_pixels"].copy() for name in labels}
            proprios = {name: arrays[name + "_state"][:4].copy() for name in labels}
        for name in labels:
            if pixels[name].shape != (224, 224, 3) or pixels[name].dtype != np.uint8:
                raise ValueError("Unexpected raw image schema")
            if proprios[name].shape != (4,) or not np.array_equal(proprios[name], proprios["original"]):
                raise ValueError("Proprio differs across marker conditions")
        if not np.array_equal(pixels["original"], pixels["restored"]):
            raise ValueError("Restoration image is not exact")

        def no_network(*args, **kwargs):
            counts["network_attempts"] += 1
            raise RuntimeError("Network/download is outside this probe")

        original_load = torch.load
        def counted_load(path, *args, **kwargs):
            name = str(Path(path).resolve())
            report["checkpoint_load_calls"][name] = report["checkpoint_load_calls"].get(name, 0) + 1
            if report["checkpoint_load_calls"][name] > 1:
                raise RuntimeError("Refuse a second load of the same checkpoint")
            return original_load(path, *args, **kwargs)

        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, MODEL, DINO_WEIGHTS
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", no_network))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", no_network))
            guards.enter_context(patch.object(torch, "load", counted_load))
            tick = time.monotonic()
            cfg = cfg_for(out, "cpu")
            model, pre, loaded = load_model(cfg, "cpu")
            report["load_seconds"] = time.monotonic() - tick
            report["strict_load"] = loaded
            report["strict_load"].pop("model_forward_calls", None)
            if set(report["checkpoint_load_calls"]) != {str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())}:
                raise RuntimeError("Unexpected checkpoint load paths")

            def forbid(name):
                def call(*args, **kwargs):
                    counts[name] += 1
                    raise RuntimeError("Forbidden operation: " + name)
                return call
            for owner, name in [(model, "unroll"), (model.model, "forward_pred"), (model.model, "rollout")]:
                guards.enter_context(patch.object(owner, name, side_effect=forbid(name)))
            model.model.predictor.register_forward_pre_hook(forbid("predictor"))
            model.model.encoder.register_forward_pre_hook(lambda *_: counts.__setitem__("visual_encoder", counts["visual_encoder"] + 1))
            model.model.proprio_encoder.register_forward_pre_hook(lambda *_: counts.__setitem__("proprio_encoder", counts["proprio_encoder"] + 1))
            original_encode = model.encode
            def encode(observation, act=True):
                if act is not True:
                    counts["goal_encode"] += 1
                    raise RuntimeError("Private target encoding is excluded")
                counts["observation_encode"] += 1
                return original_encode(observation, act=True)
            guards.enter_context(patch.object(model, "encode", side_effect=encode))
            latents, encodings = {}, {}
            for name in labels:
                observation = {"visual": torch.from_numpy(pixels[name]).permute(2, 0, 1)[None, None],
                               "proprio": torch.from_numpy(proprios[name])[None, None]}
                tick = time.monotonic()
                z_init = model.encode(observation, act=True)
                encodings[name] = {"seconds": time.monotonic() - tick}
                for key in ("visual", "proprio"):
                    value = z_init[key].detach().cpu()
                    if not bool(torch.isfinite(value).all()) or value.dtype != torch.float32:
                        raise ValueError("Native feature is nonfinite or not FP32")
                    latents[name + "_" + key] = value.numpy().copy()
                    encodings[name][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            np.savez_compressed(out / "z_init.npz", **latents)

        def compare(left, right):
            result = {}
            for key in ("visual", "proprio"):
                a, b = latents[left + "_" + key], latents[right + "_" + key]
                delta = a.astype(np.float64) - b.astype(np.float64)
                result[key] = {"exactly_equal": bool(np.array_equal(a, b)),
                    "changed_elements": int(np.count_nonzero(a != b)),
                    "max_absolute_difference": float(np.abs(delta).max()),
                    "mean_absolute_difference": float(np.abs(delta).mean()),
                    "rms_difference": float(np.sqrt(np.mean(delta * delta)))}
            return result
        report.update(encodings=encodings, goal_a_vs_goal_b=compare("goal_a", "goal_b"),
                      original_vs_restored=compare("original", "restored"),
                      input_state_origin=source["state_origin"], cuda_initialized=torch.cuda.is_initialized())
        if report["cuda_initialized"] or not all(v["exactly_equal"] for v in report["original_vs_restored"].values()):
            raise RuntimeError("CUDA or restoration control failed")
        report.update(status="completed_cpu_native_encoder_probe",
            interpretation="Paired latent dependence only. No goal decoding, classifier, incremental privacy, native query transcript, task utility or attack success measured. The changed marker is visible in the actual current observation; no private goal image/state encoding was supplied.",
            limits=["Four T=1 observation encodes, including original/restored controls, at one fixed development state.",
                    "The loader also constructs and strictly loads predictor/action components; their inference paths were blocked and never executed.",
                    "A remote latent-only forecast service is a proposed partition, not an audited deployed API.",
                    "Native CEM queries its final mean; that returned action prefix cannot be counted as information beyond the selected actions."])
    except BaseException as error:
        report.update(status="failed_closed_cpu_native_encoder_probe", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["elapsed_seconds"] = time.monotonic() - started
        persist()
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "goal_a_vs_goal_b", "original_vs_restored", "counts")}))


if __name__ == "__main__":
    main()
