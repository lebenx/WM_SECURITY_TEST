#!/usr/bin/env python3
"""Frozen cached-future 3x3 Value diagnostic; no new Future or physical task.

Run only within the frozen resource limits. All six diagonal Value outputs must reproduce
their complete cached tensors exactly before any mixed representation is scored.
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

from probe_cosmos_first_query_truth import (
    CalibrationFailure, ResourceDeferred, first_chunk, gpu_info, require,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosmos_future_modality_factorial_20261016.json"
SOURCE = ROOT / "runs/cosmos_first_query_truth_inference_20261015"
OUTPUT = ROOT / "runs/cosmos_future_modality_factorial_20261016_resume1"
SHAPE = (1, 16, 9, 28, 28)
LEVELS = ("clean", "swap", "truth")
ORDER = ("p_clean_v_clean", "p_swap_v_swap", "p_truth_v_truth",
         "p_clean_v_swap", "p_clean_v_truth", "p_swap_v_clean",
         "p_swap_v_truth", "p_truth_v_clean", "p_truth_v_swap")
BUDGET = dict(value_sampler_calls=18, denoiser_forward_calls=18, noise_calls=18,
              checkpoint_loads=2, policy_sampler_calls=0, future_sampler_calls=0,
              vae_encodes=0, vae_decodes=0, environment_steps=0,
              model_api_calls=0, training_updates=0)


def read(path):
    return json.loads(Path(path).read_text())


def main(config_path):
    protocol = read(config_path)
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2, "Frozen root changed")
    require(protocol["candidate_indices"] == [0, 1]
            and protocol["candidate_seeds"] == [31, 32], "Candidate binding changed")
    require(tuple(protocol["factor_levels"]) == LEVELS
            and tuple(protocol["execution_order"]) == ORDER, "Factorial order changed")
    conditions = {row["name"]: row for row in protocol["conditions"]}
    require(len(conditions) == 9 and conditions == {
        f"p_{p}_v_{v}": dict(name=f"p_{p}_v_{v}", P=p, V=v)
        for p in LEVELS for v in LEVELS}, "Factorial cells changed")
    require(protocol["factors"]["P"]["slots"] == [5]
            and protocol["factors"]["V"]["slots"] == [6, 7], "Slots changed")
    require(tuple(protocol["shape"]) == SHAPE and protocol["dtype"] == "torch.float32",
            "Tensor contract changed")
    require(protocol["max_activity"] == BUDGET, "Frozen activity changed")
    require((ROOT / protocol["source_dir"]).resolve() == SOURCE
            and (ROOT / protocol["source_summary"]).resolve() == SOURCE / "summary.json"
            and (ROOT / protocol["output"]).resolve() == OUTPUT, "Artifact paths changed")
    resources = protocol["resources"]
    require(resources["cuda_visible_devices"] == "0"
            and resources["minimum_free_gpu_mib"] == 8192
            and resources["cuda_memory_fraction"] == 0.28
            and resources["hard_wall_seconds"] == 120
            and resources["offload"] is False
            and resources["verify_checkpoint_hashes"] is False, "Resource contract changed")
    require(not OUTPUT.exists(), "Output exists; no overwrite or scientific retry")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "launched_config.json").write_bytes(config_path.read_bytes())
    (OUTPUT / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    counts = dict.fromkeys(BUDGET, 0)
    report = dict(status="checking_cached_sources", started_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=str(config_path.relative_to(ROOT)), source_dir=str(SOURCE.relative_to(ROOT)),
                  output=str(OUTPUT.relative_to(ROOT)), activity=counts, max_activity=BUDGET,
                  calibration=dict(all_passed=False, diagonal_all_passed_before_mixed=False, checks={}),
                  checks={}, conditions={}, ranking={}, active_call=None,
                  scope=protocol["scope"], limits=protocol["limits"], network_attempts_blocked=0)

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = OUTPUT / "summary.tmp.json"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(OUTPUT / "summary.json")

    def check(name, ok, calibration=False):
        report["checks"][name] = bool(ok)
        if calibration:
            report["calibration"]["checks"][name] = bool(ok)
        persist()
        if not ok:
            report["calibration"]["all_passed"] = False
            raise CalibrationFailure(name)

    def deadline(*_):
        raise ResourceDeferred("Frozen 120-second wall limit reached")

    def guard_deadline():
        if time.monotonic() - started >= 120:
            deadline()

    def bump(key):
        guard_deadline()
        require(counts[key] < BUDGET[key], f"Unexpected or excess {key}")
        counts[key] += 1

    def no_network(*_args, **_kwargs):
        report["network_attempts_blocked"] += 1
        raise RuntimeError("Network prohibited in cached local experiment")

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Unregistered model path: only native Value is allowed")

    old_alarm = signal.signal(signal.SIGALRM, deadline)
    old_term = signal.signal(signal.SIGTERM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    persist()
    try:
        prior = read(SOURCE / "summary.json")
        check("qualified_source", prior["status"] == "completed_calibrated_first_query_truth_countercheck"
              and prior["calibration"]["all_passed"], True)
        original_episode = read(ROOT / prior["sources"]["clean_episode"])
        check("historical_root", all(original_episode[k] == v for k, v in
              dict(task_id=3, init_id=2, seed=31, environment_seed=0, settle_steps=10).items()), True)
        historical_candidates = first_chunk(original_episode)["native_best_of_n"]["candidates"]
        utility_report = read(ROOT / protocol["historical_utility_report"])
        utility_paths = utility_report["sources"]["historical_episodes"]
        utilities = []
        for i, path in enumerate(utility_paths):
            ep = read(ROOT / path)
            expected = prior["historical_continuation_utilities"][i]
            check(f"historical_utility{i}", len(utility_paths) == 2
                  and all(ep[k] == v for k, v in dict(task_id=3, init_id=2, seed=31,
                       environment_seed=0, settle_steps=10).items())
                  and expected["candidate_index"] == i
                  and ep["success"] == expected["success"] == (i == 1)
                  and ep["environment_steps"] == expected["capped_steps"] == (300 if i == 0 else 170), True)
            utilities.append({**expected, "episode_path": path})
        require(len(utilities) == 2, "Missing historical continuation")
        report["historical_continuation_utilities"] = utilities
        report["goals"] = prior["goals"]
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["free_mib"] < 8192:
            raise ResourceDeferred("GPU0 free below8192MiB; no checkpoint loaded")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0", "Qualified physical GPU0 required")
        require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1", "Qualified torch backend required")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            guards.enter_context(patch.object(socket.socket, "connect_ex", side_effect=no_network))
            guards.enter_context(patch.object(socket, "create_connection", side_effect=no_network))
            import numpy as np
            import torch
            import cosmos_policy_native_inference as native
            check("historical_runtime", torch.__version__ == original_episode["provenance"]["torch_version"]
                  and np.__version__ == original_episode["provenance"]["numpy_version"], True)
            report["runtime"] = dict(torch_version=torch.__version__, numpy_version=np.__version__,
                                     backend="torch", device="cuda:0", computation_dtype="bfloat16",
                                     input_dtype="float32", offload=False)
            random.seed(31); np.random.seed(31); torch.manual_seed(31); torch.cuda.manual_seed_all(31)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(0.28, 0)
            torch.cuda.reset_peak_memory_stats(0)

            def cached(name):
                return torch.load(SOURCE / name, map_location="cpu", weights_only=False)

            context = cached("context.pt")
            refs = {level: [cached(f"{level}_candidate{i}.pt") if level != "truth"
                           else cached(f"truth_candidate{i}_value.pt") for i in (0, 1)] for level in LEVELS}
            sources = {level: [cached(f"truth_candidate{i}_input.pt")["value_input"] if level == "truth"
                               else refs[level][i]["future_latent"] for i in (0, 1)] for level in LEVELS}
            source_before = {level: [x.clone() for x in values] for level, values in sources.items()}
            noise = {seed: cached(f"noise_seed{seed}.pt")["unit_noise"] for seed in (31, 32)}
            original_text = context["original_text"].clone()
            check("original_text_shape_dtype", tuple(original_text.shape) == (1, 512, 1024)
                  and original_text.dtype == torch.bfloat16 and torch.isfinite(original_text).all(), True)
            for i in (0, 1):
                clean = refs["clean"][i]
                check(f"candidate{i}_action_binding", torch.equal(clean["action_latent"], refs["swap"][i]["action_latent"])
                      and torch.equal(clean["action_latent"][:, :, :4], context["observed_latent"][:, :, :4])
                      and np.array_equal(clean["environment_actions"], np.asarray(historical_candidates[i]["environment_actions"], dtype=np.float64))
                      and historical_candidates[i]["seed"] == 31 + i, True)
                for level in LEVELS:
                    x, ref = sources[level][i], refs[level][i]
                    check(f"{level}{i}_source_contract", tuple(x.shape) == SHAPE
                          and x.dtype == torch.float32 and torch.isfinite(x).all()
                          and torch.equal(x[:, :, :5], clean["action_latent"][:, :, :5])
                          and ref["seed"] == 31 + i and tuple(ref["value_latent"].shape) == SHAPE
                          and torch.isfinite(ref["value_latent"]).all(), True)
                    if level == "truth":
                        check(f"truth{i}_reference_input", torch.equal(x, ref["value_input"]), True)
                    score_key = "true_future" if level == "truth" else level
                    check(f"{level}{i}_cached_scalar", float(ref["value"].item()) == prior["scores"][score_key][i], True)
            original_load, loaded = torch.load, []

            def load_once(path, *args, **kwargs):
                path = Path(path).resolve()
                require(path in (native.MODEL, native.TOKENIZER) and path not in loaded, "Unexpected checkpoint load")
                bump("checkpoint_loads"); loaded.append(path)
                return original_load(str(path), *args, **kwargs)

            for attr in ("get_action", "get_future_state_prediction"):
                guards.enter_context(patch.object(native, attr, side_effect=forbidden))
            for attr in ("encode", "decode"):
                guards.enter_context(patch.object(native.OfficialWanVAE, attr, side_effect=forbidden))
            with torch.inference_mode():
                report["status"] = "loading_existing_checkpoint"; persist()
                load_started = time.monotonic()
                with patch.object(torch, "load", side_effect=load_once):
                    model, vae, sampler, base_rand = native.load_model(device="cuda", verify=False)
                del vae
                torch.cuda.synchronize(); guard_deadline()
                report["checkpoint_loading_seconds"] = time.monotonic() - load_started
                report["strict_load"] = model.checkpoint_adapter_metadata
                report["checkpoint_paths"] = [str(p.relative_to(ROOT)) for p in loaded]
                require(counts["checkpoint_loads"] == 2 and not model.training, "Unexpected loaded model")
                text = original_text.to(device="cuda", dtype=torch.bfloat16)

                def record_forward(_module, _args):
                    require(report["active_call"] is not None, "Denoiser outside registered Value call")
                    bump("denoiser_forward_calls")

                guards.callback(model.register_forward_pre_hook(record_forward).remove)

                def rand(shape, seed):
                    active = report["active_call"]
                    require(tuple(shape) == SHAPE and active is not None and seed == active["seed"],
                            "Noise shape/candidate binding changed")
                    bump("noise_calls")
                    value = base_rand(shape, seed=seed)
                    check(active["key"] + "_noise_exact", torch.equal(value.cpu(), noise[seed]))
                    return value

                for condition_index, name in enumerate(ORDER):
                    diagonal = condition_index < 3
                    if not diagonal:
                        require(report["calibration"]["diagonal_all_passed_before_mixed"], "Mixed call before calibration")
                    cell = conditions[name]
                    report["status"] = "calibrating_diagonals" if diagonal else "running_frozen_mixed_cells"
                    report["conditions"][name] = []; persist()
                    for i in (0, 1):
                        key = f"{name}_candidate{i}"
                        request_cpu = sources["clean"][i].clone()
                        request_cpu[:, :, 5:6] = sources[cell["P"]][i][:, :, 5:6]
                        request_cpu[:, :, 6:8] = sources[cell["V"]][i][:, :, 6:8]
                        check(key + "_construction_exact", torch.equal(request_cpu[:, :, :5], sources["clean"][i][:, :, :5])
                              and torch.equal(request_cpu[:, :, 8:], sources["clean"][i][:, :, 8:])
                              and torch.equal(request_cpu[:, :, 5:6], sources[cell["P"]][i][:, :, 5:6])
                              and torch.equal(request_cpu[:, :, 6:8], sources[cell["V"]][i][:, :, 6:8]))
                        request = request_cpu.to("cuda")
                        before, text_before = request.clone(), text.clone()
                        report["active_call"] = dict(key=key, condition=name, candidate_index=i, seed=31+i)
                        bump("value_sampler_calls"); persist()
                        call_counts = dict(counts)
                        torch.cuda.synchronize(); tick = time.monotonic()
                        value, value_latent = native.get_value_prediction(model, sampler, rand, request, text, 31+i, 1)
                        torch.cuda.synchronize(); guard_deadline()
                        seconds = time.monotonic() - tick
                        output, scalar = value_latent.detach().cpu(), value.detach().cpu()
                        check(key + "_value_output", tuple(output.shape) == SHAPE and output.dtype == torch.float32
                              and tuple(scalar.shape) == (1,) and torch.isfinite(output).all() and torch.isfinite(scalar).all())
                        check(key + "_input_and_text_unchanged", torch.equal(request, before)
                              and torch.equal(text, text_before) and torch.equal(text.cpu(), original_text))
                        check(key + "_conditioned_prefix_exact", torch.equal(output[:, :, :8], request_cpu[:, :, :8]))
                        check(key + "_one_value_call", counts["noise_calls"] == call_counts["noise_calls"] + 1
                              and counts["denoiser_forward_calls"] == call_counts["denoiser_forward_calls"] + 1)
                        for level in LEVELS:
                            check(key + f"_{level}_sources_unchanged", all(torch.equal(a, b) for a, b in
                                  zip(sources[level], source_before[level])))
                        artifact = key + ".pt"
                        torch.save(dict(value_input=request_cpu, value_latent=output, value=scalar,
                                        candidate_index=i, seed=31+i, P=cell["P"], V=cell["V"]), OUTPUT / artifact)
                        delta = request_cpu[:, :, 5:8].double() - sources["truth"][i][:, :, 5:8].double()
                        row = dict(candidate_index=i, seed=31+i, P=cell["P"], V=cell["V"], value=float(scalar.item()),
                                   future_proprio_slot_mse=float(delta[:, :, :1].square().mean()),
                                   visual_latent_mse=float(delta[:, :, 1:].square().mean()),
                                   future_wrist_slot_mse=float(delta[:, :, 1:2].square().mean()),
                                   future_primary_slot_mse=float(delta[:, :, 2:3].square().mean()),
                                   all_future_slots_mse=float(delta.square().mean()), historical_utility=utilities[i],
                                   tensor_file=artifact, value_call_index=counts["value_sampler_calls"], elapsed_seconds=seconds)
                        report["conditions"][name].append(row); persist()
                        if diagonal:
                            ref = refs[cell["P"]][i]
                            check(key + "_full_value_exact", torch.equal(output, ref["value_latent"]), True)
                            check(key + "_scalar_exact", torch.equal(scalar, ref["value"]), True)
                        report["active_call"] = None; persist()
                    scores = [row["value"] for row in report["conditions"][name]]
                    selected = 0 if scores[0] >= scores[1] else 1
                    report["ranking"][name] = dict(selected_index=selected, exact_tie=scores[0] == scores[1],
                        candidate1_minus_candidate0_value=scores[1]-scores[0], historical_selected_utility=utilities[selected])
                    if condition_index == 2:
                        require(counts["value_sampler_calls"] == 6, "Diagonal activity changed")
                        report["calibration"].update(all_passed=True, diagonal_all_passed_before_mixed=True)
                    persist()
                require(counts == BUDGET, "Final activity differs from frozen budget")
                guard_deadline()
                report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
                report["status"] = "completed_calibrated_future_modality_factorial"
                report["completed_at_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    except BaseException as error:
        report["status"] = ("resource_deferred" if isinstance(error, ResourceDeferred) else
                            "calibration_failed" if isinstance(error, CalibrationFailure) else "error")
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        persist()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    main(args.config.resolve())
