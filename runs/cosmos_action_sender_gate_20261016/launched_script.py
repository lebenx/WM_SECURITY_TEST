#!/usr/bin/env python3
"""Fixed-root full-layer sender K/V donor diagnostic; no new task episodes.

Only Future is instrumented. Clean/Swap same-candidate baselines supply donor
K/V after native normalization/RoPE. Non-target rows retain the current hybrid
recipient, not a frozen baseline trajectory. No API, training or seed search.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
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
CONFIG = ROOT / "configs/cosmos_action_sender_gate_20261016.json"
SOURCE = ROOT / "runs/cosmos_first_query_truth_inference_20261015"
SHAPE = (1, 16, 9, 28, 28)
CONDITIONS = (
    ("clean", "clean", None, None),
    ("swap", "swap", None, None),
    ("clean_action_swap", "clean", "swap", 4),
    ("swap_action_clean", "swap", "clean", 4),
    ("clean_proprio_swap", "clean", "swap", 1),
    ("swap_proprio_clean", "swap", "clean", 1),
)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def sender_kv(model, torch, diagnostics, *, capture=None, donor=None, frame=None):
    """Capture or replace sender K/V after native normalization and RoPE."""
    require((capture is not None) != (donor is not None), "Choose capture or donor")
    require(capture is not None or frame in (1, 4), "Unregistered sender group")
    require(len(model.blocks) == 28, "Unexpected block count")
    diagnostics.update(mode="capture" if capture is not None else "replace", frame=frame,
                       rows_per_block=196, calls_by_block=[0] * 28,
                       nonselected_rows_exact=True, q_exact=True, native_kv_unmutated=True,
                       rng_unchanged=True, replacement_squared_sum=0.0,
                       replacement_elements=0, layer_deltas=[])
    with ExitStack() as stack:
        for index, block in enumerate(model.blocks):
            attention = block.self_attn
            require(attention.is_selfattn and attention.backend == "torch"
                    and attention.qkv_format == "bshd" and block.cp_size in (None, 1),
                    "Unexpected attention/layout/parallel path")
            require(not attention.training and isinstance(attention.output_dropout, torch.nn.Identity)
                    and attention.n_heads * attention.head_dim == 2048, "Stochastic or wrong attention")
            original = attention.compute_attention
            expected = (1, 1764, attention.n_heads, attention.head_dim)

            def wrapped(q, k, v, video_size=None, kv_cache_cfg=None,
                        _forward=original, _index=index, _expected=expected):
                require(kv_cache_cfg is None, "Unexpected KV cache")
                require(video_size is not None and (video_size.T, video_size.H, video_size.W) == (9, 14, 14),
                        "Unexpected video token grid")
                require(diagnostics["calls_by_block"][_index] == 0, "Multiple denoiser invocations")
                require(all(tuple(t.shape) == _expected and torch.isfinite(t).all() for t in (q, k, v)),
                        "Unexpected or nonfinite QKV")
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                q_before, k_before, v_before = q.clone(), k.clone(), v.clone()
                k_out, v_out = k, v
                if capture is not None:
                    require(_index not in capture, "Duplicate donor layer")
                    capture[_index] = {f: {"k": k[:, f * 196:(f + 1) * 196].cpu().clone(),
                                          "v": v[:, f * 196:(f + 1) * 196].cpu().clone()}
                                       for f in (1, 4)}
                    if _index == 0:
                        capture[_index]["first_qkv"] = [t.cpu().clone() for t in (q, k, v)]
                else:
                    require(set(donor) == set(range(28)), "Incomplete donor")
                    start, stop = frame * 196, (frame + 1) * 196
                    k_out, v_out = k.clone(), v.clone()
                    layer = {"block": _index, "invocation": 1}
                    for kind, base, copied in (("k", k, k_out), ("v", v, v_out)):
                        saved = donor[_index][frame][kind]
                        require(saved.shape == base[:, start:stop].shape and saved.dtype == base.dtype
                                and torch.isfinite(saved).all(), "Wrong donor shape/dtype")
                        replacement = saved.to(device=base.device)
                        copied[:, start:stop] = replacement
                        require(torch.equal(copied[:, :start], base[:, :start])
                                and torch.equal(copied[:, stop:], base[:, stop:])
                                and torch.equal(copied[:, start:stop], replacement), "Sender copy boundary failed")
                        delta = replacement.double() - base[:, start:stop].double()
                        squared = float(delta.square().sum())
                        require(_index != 0 or squared == 0.0, "Initial layer changed before text entry")
                        layer[kind] = {"elements": delta.numel(), "max_abs": float(delta.abs().max()),
                                       "l2": squared ** 0.5, "rms": (squared / delta.numel()) ** 0.5}
                        diagnostics["replacement_squared_sum"] += squared
                        diagnostics["replacement_elements"] += delta.numel()
                    diagnostics["layer_deltas"].append(layer)
                result = _forward(q, k_out, v_out, video_size=video_size, kv_cache_cfg=kv_cache_cfg)
                require(torch.equal(q, q_before) and torch.equal(k, k_before)
                        and torch.equal(v, v_before), "Native QKV mutated")
                require(torch.equal(cpu_rng, torch.get_rng_state())
                        and torch.equal(cuda_rng, torch.cuda.get_rng_state()), "Hook consumed RNG")
                diagnostics["calls_by_block"][_index] += 1
                return result

            stack.enter_context(patch.object(attention, "compute_attention", new=wrapped))
        yield
    require(diagnostics["calls_by_block"] == [1] * 28, "Missing or repeated layer")
    diagnostics["replacement_rms"] = ((diagnostics["replacement_squared_sum"] /
                                       diagnostics["replacement_elements"]) ** 0.5
                                      if diagnostics["replacement_elements"] else None)


def main(config_path):
    protocol = read(config_path)
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2, "Frozen root changed")
    require(protocol["candidate_seeds"] == [31, 32], "Frozen candidates changed")
    require(protocol["max_activity"]["future_sampler_calls"] == 14
            and protocol["max_activity"]["value_sampler_calls"] == 14, "Call budget changed")
    require(protocol["source"] == str(SOURCE.relative_to(ROOT)), "Source changed")
    require(protocol["minimum_free_gpu_mib"] == 8192 and protocol["cuda_memory_fraction"] == 0.28
            and protocol["hard_wall_seconds"] == 120, "Resource constraints changed")
    require(protocol["experiment"] == "cosmos_action_sender_kv_donor_gate"
            and protocol["future_steps"] == protocol["value_steps"] == 1, "Experiment changed")
    prior = read(SOURCE / "summary.json")
    require(prior["status"] == "completed_calibrated_first_query_truth_countercheck"
            and prior["calibration"]["all_passed"], "Cached experiment is not qualified")
    require([x["success"] for x in prior["historical_continuation_utilities"]] == [False, True]
            and [x["capped_steps"] for x in prior["historical_continuation_utilities"]] == [300, 170],
            "Cached utility binding changed")
    out = ROOT / protocol["output"]
    require(not out.exists(), "Output exists; no overwrite or scientific retry")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    counts = {k: 0 for k in protocol["max_activity"]}
    report = {"status": "checking_resources", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path.relative_to(ROOT)), "source": str(SOURCE.relative_to(ROOT)),
              "activity": counts, "calibration": {"all_passed": False, "checks": {}},
              "conditions": {}, "noops": {},
              "historical_continuation_utilities": prior["historical_continuation_utilities"],
              "scope": "One post-hoc known harmful root. Researcher internal activation intervention. Existing actual continuation outcomes; no new physical recovery or deployable defense."}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = out / "summary.tmp.json"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "summary.json")

    def check(name, ok):
        report["calibration"]["checks"][name] = bool(ok)
        persist()
        if not ok:
            raise CalibrationFailure(name)

    def deadline(*_):
        raise ResourceDeferred("Registered 120-second deadline reached")

    def no_network(*_args, **_kwargs):
        raise RuntimeError("Network prohibited in this local experiment")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    persist()
    try:
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["free_mib"] < 8192:
            raise ResourceDeferred("GPU0 free below8192MiB; no checkpoint loaded")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0", "Qualified GPU0 required")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1", "Qualified torch backend required")
        import numpy as np
        import torch
        import cosmos_policy_native_inference as native
        original_episode = read(ROOT / prior["sources"]["clean_episode"])
        require(torch.__version__ == original_episode["provenance"]["torch_version"]
                and np.__version__ == original_episode["provenance"]["numpy_version"], "Runtime changed")
        random.seed(31); np.random.seed(31); torch.manual_seed(31); torch.cuda.manual_seed_all(31)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.28, 0)
        torch.cuda.reset_peak_memory_stats(0)
        context = torch.load(SOURCE / "context.pt", map_location="cpu", weights_only=False)
        references = {name: [torch.load(SOURCE / f"{name}_candidate{i}.pt", map_location="cpu",
                                       weights_only=False) for i in range(2)] for name in ("clean", "swap")}
        truth = [torch.load(SOURCE / f"truth_candidate{i}_input.pt", map_location="cpu", weights_only=False)
                 for i in range(2)]
        noise = {seed: torch.load(SOURCE / f"noise_seed{seed}.pt", map_location="cpu", weights_only=False)
                 ["unit_noise"] for seed in (31, 32)}
        text = {"clean": context["original_text"].to(device="cuda", dtype=torch.bfloat16),
                "swap": context["swap_text"].to(device="cuda", dtype=torch.bfloat16)}
        text = {k: v[None] if v.ndim == 2 else v for k, v in text.items()}
        actions = [references["clean"][i]["action_latent"].cuda() for i in range(2)]
        for i in range(2):
            require(tuple(actions[i].shape) == SHAPE
                    and torch.equal(actions[i][:, :, :4].cpu(), context["observed_latent"][:, :, :4]),
                    "Wrong cached candidate context")
            require(torch.equal(actions[i].cpu(), references["swap"][i]["action_latent"]),
                    "Candidate action differs between backgrounds")
        for i in range(2):
            require(references["clean"][i]["seed"] == references["swap"][i]["seed"] == 31 + i,
                    "Cached donor candidate seed changed")
            require(torch.equal(native.action_from_latent(actions[i]).cpu(),
                                references["clean"][i]["normalized_actions"]),
                    "Cached action decoding differs")
            require(prior["calibration"]["checks"][f"clean{i}_controls_exact"]
                    and prior["calibration"]["checks"][f"swap{i}_controls_exact"]
                    and np.isfinite(references["clean"][i]["environment_actions"]).all(),
                    "Historical physical controls not calibrated")
        original_load = torch.load
        loaded = []

        def load_once(path, *args, **kwargs):
            p = Path(path).resolve()
            require(p in (native.MODEL, native.TOKENIZER) and p not in loaded, "Unexpected model load")
            loaded.append(p); counts["checkpoint_loads"] += 1
            return original_load(path, *args, **kwargs)

        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            report["status"] = "loading_existing_checkpoint"; persist()
            with patch.object(torch, "load", side_effect=load_once):
                model, vae, sampler, base_rand = native.load_model(device="cuda", verify=False)
            del vae
            report["strict_load"] = model.checkpoint_adapter_metadata
            require(model.patch_spatial == 2 and model.patch_temporal == 1,
                    "Model token grid changed")
            donors = {name: [{}, {}] for name in ("clean", "swap")}
            native_attention_methods = [b.self_attn.compute_attention for b in model.blocks]
            stage = {"kind": None}
            report["denoiser_forward_calls"] = {"future": 0, "value": 0}

            def record_forward(_module, _args, _output):
                require(stage["kind"] in ("future", "value"), "Unregistered denoiser call")
                report["denoiser_forward_calls"][stage["kind"]] += 1

            guards.callback(model.register_forward_hook(record_forward).remove)

            def rand(shape, seed):
                require(tuple(shape) == SHAPE and seed in noise, "Noise request changed")
                counts["noise_calls"] += 1
                value = base_rand(shape, seed=seed)
                require(torch.equal(value.cpu(), noise[seed]), "Historical seed-local noise changed")
                return value

            def sample(name, i, background, alternate=None, frame=None, capture=False):
                for key in ("future_sampler_calls", "value_sampler_calls"):
                    require(counts[key] < protocol["max_activity"][key], "Sampler budget exhausted")
                action_before = actions[i].clone()
                text_before = {k: v.clone() for k, v in text.items()}
                diagnostic = {}
                forwards_before = report["denoiser_forward_calls"].copy()
                counts["future_sampler_calls"] += 1
                stage["kind"] = "future"
                with ExitStack() as hooks:
                    if capture:
                        hooks.enter_context(sender_kv(model, torch, diagnostic, capture=donors[background][i]))
                    elif frame is not None:
                        hooks.enter_context(sender_kv(model, torch, diagnostic, donor=donors[alternate][i], frame=frame))
                    future = native.get_future_state_prediction(model, sampler, rand, actions[i],
                                                               text[background], 31 + i, 1)
                require(report["denoiser_forward_calls"]["future"] == forwards_before["future"] + 1,
                        "Future did not use exactly one denoiser evaluation")
                require(all(b.self_attn.compute_attention == method for b, method in
                            zip(model.blocks, native_attention_methods)), "Sender hook leaked into Value")
                # All hooks have been removed before the original-goal Value call.
                counts["value_sampler_calls"] += 1
                stage["kind"] = "value"
                hook_count = list(diagnostic.get("calls_by_block", []))
                value, value_latent = native.get_value_prediction(model, sampler, rand, future, text["clean"], 31 + i, 1)
                require(report["denoiser_forward_calls"]["value"] == forwards_before["value"] + 1,
                        "Value did not use exactly one denoiser evaluation")
                stage["kind"] = None
                require(hook_count == diagnostic.get("calls_by_block", []), "Route hook leaked into Value")
                require(torch.equal(actions[i], action_before)
                        and all(torch.equal(text[k], text_before[k]) for k in text), "Protected input mutated")
                require(torch.equal(future[:, :, :5], actions[i][:, :, :5].float()), "Physical action prefix changed")
                require(torch.isfinite(future).all() and torch.isfinite(value_latent).all()
                        and torch.isfinite(value).all(), "Nonfinite sampler output")
                cached = future.cpu()
                delta = cached[:, :, 6:8].double() - truth[i]["value_input"][:, :, 6:8].double()
                row = {"candidate_index": i, "value": float(value.item()),
                       "visual_latent_mse": float(delta.square().mean()), "hook": diagnostic,
                       "future_file": f"{name}_candidate{i}.pt"}
                value_cached = value_latent.cpu()
                torch.save({"future_latent": cached, "value_latent": value_cached,
                            "value": row["value"], "candidate_index": i,
                            "background": background, "alternate": alternate, "frame": frame},
                           out / row["future_file"])
                return row, cached, value_cached

            report["status"] = "calibrating_both_backgrounds"; persist()
            for name, background, alternate, frame in CONDITIONS[:2]:
                report["conditions"][name] = []
                for i in range(2):
                    row, future, value_latent = sample(name, i, background, capture=True)
                    torch.save(donors[background][i], out / f"donor_{background}_candidate{i}.pt")
                    check(f"{name}{i}_future_exact", torch.equal(future, references[name][i]["future_latent"]))
                    check(f"{name}{i}_value_latent_exact", torch.equal(value_latent, references[name][i]["value_latent"]))
                    check(f"{name}{i}_value_exact", row["value"] == prior["scores"][name][i])
                    report["conditions"][name].append(row)
            for i in range(2):
                check(f"candidate{i}_first_layer_qkv_goal_independent",
                      all(torch.equal(a, b) for a, b in zip(donors["clean"][i][0]["first_qkv"],
                                                          donors["swap"][i][0]["first_qkv"])))
            report["status"] = "calibrating_identity_senders"; persist()
            for background, i, frame in (("clean", 0, 4), ("swap", 1, 1)):
                name = f"noop_{background}_frame{frame}"
                row, future, value_latent = sample(name, i, background, background, frame)
                check(name + "_future_exact", torch.equal(future, references[background][i]["future_latent"]))
                check(name + "_value_latent_exact", torch.equal(value_latent, references[background][i]["value_latent"]))
                check(name + "_value_exact", row["value"] == prior["scores"][background][i])
                check(name + "_replacement_zero", row["hook"]["replacement_squared_sum"] == 0.0)
                report["noops"][name] = row
            report["calibration"]["all_passed"] = True
            report["status"] = "running_four_frozen_sender_arms"; persist()
            for name, background, alternate, frame in CONDITIONS[2:]:
                report["conditions"][name] = []
                for i in range(2):
                    row, _, _ = sample(name, i, background, alternate, frame)
                    report["conditions"][name].append(row); persist()
            require(counts == protocol["max_activity"], "Final activity differs from frozen budget")
            require(report["denoiser_forward_calls"] == {"future": 14, "value": 14},
                    "Unexpected denoiser forward count")
            report["extra_attention_forwards"] = 0
            report["sender_hook_calls"] = sum(sum(r["hook"]["calls_by_block"])
                for rows in report["conditions"].values() for r in rows) + sum(
                sum(r["hook"]["calls_by_block"]) for r in report["noops"].values())
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            report["status"] = "completed_calibrated_bidirectional_sender_kv"
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
