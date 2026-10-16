#!/usr/bin/env python3
"""Frozen finite-bank feasibility check on existing Cosmos candidate tensors.

This changes the native cross-stage seed protocol. Only the marginal of each
new finite bank is preserved by the subsequent coupling calculation. No new
action, environment transition, model update or API request is made.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import socket
import time
import traceback
from unittest.mock import patch

from probe_cosmos_first_query_truth import gpu_info, require, ResourceDeferred, CalibrationFailure

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosmos_forecast_coupling_gate_20261015.json"
SHAPE = (1, 16, 9, 28, 28)


def read(path):
    return json.loads(Path(path).read_text())


def main(config_path):
    protocol = read(config_path)
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2, "Frozen root changed")
    require(protocol["candidate_indices"] == [0, 1] and protocol["candidate_seeds"] == [31, 32]
            and protocol["value_seeds"] == [31, 32], "Frozen actions/Value noise changed")
    require(protocol["forecast_seeds"] == list(range(401, 409)), "Frozen bank seeds changed")
    require(protocol["future_steps"] == protocol["value_steps"] == 1, "Native steps changed")
    require(protocol["minimum_free_gpu_mib"] == 8192 and protocol["cuda_memory_fraction"] == 0.28
            and protocol["hard_wall_seconds"] == 120, "Resource gate changed")
    source = ROOT / protocol["source"]
    require(source == ROOT / "runs/cosmos_first_query_truth_inference_20261015", "Source changed")
    prior = read(source / "summary.json")
    require(prior["status"] == "completed_calibrated_first_query_truth_countercheck"
            and prior["calibration"]["all_passed"], "Prior query is not qualified")
    require([x["success"] for x in prior["historical_continuation_utilities"]] == [False, True]
            and [x["capped_steps"] for x in prior["historical_continuation_utilities"]] == [300, 170],
            "Historical utility binding changed")
    out = ROOT / protocol["output"]
    require(not out.exists(), "Output already exists; no overwrite or scientific retry")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    counts = {key: 0 for key in protocol["max_activity"]}
    report = {
        "status": "checking_resources", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(config_path.relative_to(ROOT)), "source": protocol["source"],
        "activity": counts, "calibration": {"all_passed": False, "checks": {}},
        "forecast_seeds": protocol["forecast_seeds"], "value_seeds": [31, 32],
        "scores_by_candidate": [[], []], "visual_mse_by_candidate": [[], []],
        "historical_continuation_utilities": prior["historical_continuation_utilities"],
        "native_vs_added": protocol["native_vs_added"],
        "scope": "One posthoc root, added finite-bank protocol, existing real branch utility; no new physical execution.",
        "limits": ["Not a preservation claim about the original native cross-stage seed law.",
                   "Finite-bank marginal preservation is not preservation of the full diffusion distribution.",
                   "An oracle optimal pairing is only an upper bound on a restricted sampler threat.",
                   "Bank rows and matrix entries are not independent task replications."]
    }

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        tmp = out / "summary.tmp.json"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "summary.json")

    def deadline(*_):
        raise ResourceDeferred("120-second process deadline reached")

    def no_network(*_args, **_kwargs):
        raise RuntimeError("Network forbidden in this local experiment")

    def check(name, passed):
        report["calibration"]["checks"][name] = bool(passed)
        persist()
        if not passed:
            raise CalibrationFailure(name)

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    persist()
    try:
        report["gpu_at_start"] = gpu_info()
        persist()
        if report["gpu_at_start"]["free_mib"] < 8192:
            raise ResourceDeferred("GPU0 has less than8192MiB free; no model load")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0", "GPU0 required")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1", "Use qualified torch backend")
        import numpy as np
        import torch
        import cosmos_policy_native_inference as native
        original = read(ROOT / prior["sources"]["clean_episode"])
        require(torch.__version__ == original["provenance"]["torch_version"]
                and np.__version__ == original["provenance"]["numpy_version"], "Qualified runtime changed")
        random.seed(31); np.random.seed(31); torch.manual_seed(31); torch.cuda.manual_seed_all(31)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.28, 0)
        torch.cuda.reset_peak_memory_stats(0)
        context = torch.load(source / "context.pt", map_location="cpu", weights_only=False)
        clean = [torch.load(source / f"clean_candidate{i}.pt", map_location="cpu", weights_only=False)
                 for i in range(2)]
        truth = [torch.load(source / f"truth_candidate{i}_input.pt", map_location="cpu", weights_only=False)
                 for i in range(2)]
        text = context["original_text"].cuda()
        actions = [x["action_latent"].cuda() for x in clean]
        require(all(tuple(x.shape) == SHAPE for x in actions), "Wrong action latent shape")
        for i in range(2):
            require(torch.equal(actions[i][:, :, :4].cpu(), context["observed_latent"][:, :, :4]),
                    "Cached action belongs to a different observation")
            require(clean[i]["seed"] == 31 + i and float(clean[i]["value"].item()) == prior["scores"]["clean"][i],
                    "Cached original candidate/score mismatch")
        original_load = torch.load
        loaded = []

        def load_once(path, *args, **kwargs):
            p = Path(path).resolve()
            require(p in (native.MODEL, native.TOKENIZER) and p not in loaded,
                    "Unexpected/repeated model load")
            loaded.append(p)
            counts["checkpoint_loads"] += 1
            return original_load(path, *args, **kwargs)

        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            report["status"] = "loading_existing_checkpoint"; persist()
            with patch.object(torch, "load", side_effect=load_once):
                model, vae, sampler, rand = native.load_model(device="cuda", verify=False)
            del vae
            require(len(loaded) == 2, "Incomplete model load")
            report["strict_load"] = model.checkpoint_adapter_metadata

            def sample(i, future_seed):
                before = actions[i].clone()
                text_before = text.clone()
                for kind in ("future_sampler_calls", "value_sampler_calls"):
                    require(counts[kind] < protocol["max_activity"][kind], "Model call budget exhausted")
                counts["future_sampler_calls"] += 1
                future = native.get_future_state_prediction(model, sampler, rand, actions[i], text,
                                                           future_seed, 1)
                counts["value_sampler_calls"] += 1
                value, _ = native.get_value_prediction(model, sampler, rand, future, text, 31 + i, 1)
                require(torch.equal(actions[i], before) and torch.equal(text, text_before), "Protected input mutated")
                require(torch.equal(future[:, :, :5], actions[i][:, :, :5].float()), "Forecast changed fixed prefix")
                return future.cpu(), float(value.item())

            report["status"] = "calibrating_cached_clean"; persist()
            for i in range(2):
                future, score = sample(i, 31 + i)
                check(f"candidate{i}_future_exact", torch.equal(future, clean[i]["future_latent"]))
                check(f"candidate{i}_value_exact", score == prior["scores"]["clean"][i])
            report["calibration"]["all_passed"] = True
            report["status"] = "generating_frozen_bank"; persist()
            for i in range(2):
                bank = []
                for seed in protocol["forecast_seeds"]:
                    future, score = sample(i, seed)
                    bank.append(future)
                    delta = future[:, :, 6:8].double() - truth[i]["value_input"][:, :, 6:8].double()
                    report["scores_by_candidate"][i].append(score)
                    report["visual_mse_by_candidate"][i].append(float(delta.square().mean()))
                    persist()
                torch.save({"forecast_seeds": protocol["forecast_seeds"], "candidate_index": i,
                            "future_latents": torch.stack(bank), "scores": report["scores_by_candidate"][i]},
                           out / f"candidate{i}_bank.pt")
            require(counts == protocol["max_activity"], "Final execution budget differs from protocol")
            report["mean_visual_mse_by_candidate"] = [sum(x) / len(x) for x in report["visual_mse_by_candidate"]]
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            report["status"] = "completed_frozen_forecast_bank"
            persist()
    except BaseException as error:
        report["status"] = ("resource_deferred" if isinstance(error, ResourceDeferred) else
                            "calibration_failed" if isinstance(error, CalibrationFailure) else "error")
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        persist()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    main(args.config.resolve())
