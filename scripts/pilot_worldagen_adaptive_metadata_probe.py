#!/usr/bin/env python3
"""Bounded, prefix-only metadata-selection probe; never creates an environment.

The sole write delivered to a hypothetical contributor is a 30-window sequence
of two registered goal-token arrays. Physical observations/actions are immutable.
This reconstructs the existing LIBERO forecast-LoRA recipe, not an unavailable
official LIBERO trainer. BC is an attack surrogate, not measured task utility.

signal/collect/verify load the frozen model only when explicitly invoked. select
and --self-test are CPU-only. Every stage binds its protocol, input and sources;
completed stages and per-window collection checkpoints can be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import importlib.metadata
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import pilot_worldagen_forecast_ttt_task as core

ROOT = Path(__file__).resolve().parents[1]
STARTS = list(range(0, 180, 6))
FRAMES = list(range(0, 181, 3))
CONTRACT = {
    "adapter_seed": 42, "windows": 30, "world_loss_scale": .1,
    "lr": 5e-4, "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8,
    "clip_norm": .1, "optimizer_updates_per_assignment": 1,
    "surrogate": "mean(SmoothL1(arm,beta=1)+0.01*BCE(gripper))",
    "surrogate_rng": "20261014+window_start; model.eval; original Goal0",
    "collection_rng": "seed42 once; same saved RNG before either goal at each window",
    "selector_seed": 314159, "selector": "one lexicographic i<j sweep; skip equal labels; strictly improving swaps only",
    "goal_counts": [15, 15], "metadata_poison_fraction_vs_native_no_lang": 1.0,
    "restarts": 0, "heldout_access": False,
    "task_evaluation": False, "attack_surrogate_is_task_utility": False,
}


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": core.file_sha(path)}


def check_artifact(item):
    path = Path(item["path"])
    if core.file_sha(path) != item["sha256"]:
        raise ValueError(f"Artifact changed: {path}")
    return path


def save_tensor(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return artifact(path)


def read_tensor(item):
    return torch.load(check_artifact(item), map_location="cpu", weights_only=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_prefix(cache):
    # Never accept the original cache, even if a caller promises not to use its
    # heldout rows. Selection only receives a physically truncated copy.
    allowed = {"images", "states", "actions", "frame_indices", "train_starts"}
    require(set(cache) == allowed, f"Require exactly five prefix fields: {set(cache)^allowed}")
    require(cache["train_starts"] == STARTS, "Require fixed starts 0..174 stride6")
    require(cache["frame_indices"] == FRAMES, "Require only frames 0..180 stride3")
    require(tuple(cache["images"].shape) == (61, 2, 3, 224, 224), "Prefix image shape mismatch")
    require(tuple(cache["states"].shape) == (61, 7), "Prefix state shape mismatch")
    require(tuple(cache["actions"].shape) == (180, 7), "Require exactly 180 controls")
    for key in ("images", "states", "actions"):
        require(cache[key].device.type == "cpu", "Prefix must be stored on CPU")
        require(cache[key].dtype == torch.float32, f"Unexpected {key} dtype")
        require(bool(torch.isfinite(cache[key]).all()), f"Nonfinite {key}")
    require(bool(((cache["actions"][:, 6] == -1) | (cache["actions"][:, 6] == 1)).all()),
            "Expected recorded binary gripper commands")


def prepare(args):
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("adapter_seed") == 42, "Protocol adapter_seed must be42")
    goals = protocol["goals"]
    require(set(goals) == {"goal0", "goal1"}, "Require exactly two fixed goals")
    require(all(isinstance(x, str) and x for x in goals.values()), "Empty goal")
    require(goals["goal0"] != goals["goal1"], "Goals must differ")
    prefix_record = protocol["prefix_cache"]
    prefix = check_artifact(prefix_record)
    registered_initial = check_artifact(protocol["initial_adapter"])
    if args.prefix_cache is not None:
        require(args.prefix_cache.resolve() == prefix.resolve(), "CLI prefix differs from registration")
    sources = protocol.get("source_files", {})
    require(bool(sources), "Registration must bind source_files")
    for filename, expected in sources.items():
        require(core.file_sha(ROOT / filename) == expected, f"Source changed: {filename}")
    implementation = [Path(__file__), Path(core.__file__), ROOT / "scripts/pilot_worldagen_libero_closed_loop.py",
                      ROOT / "vendor/WorldAgen/models/model.py", ROOT / "vendor/WorldAgen/models/qwen3.py"]
    require(str(Path(__file__).relative_to(ROOT)) in sources, "Registration must bind this probe's SHA")
    assets = [ROOT / "models/WorldAgen" / filename for filename in
              ("libero-38.pth", "ViT-B-32.pt", "mae_pretrain_vit_base.pth", "provenance.json")]
    asset_records = [artifact(p) for p in assets]
    provenance = json.loads(assets[-1].read_text())
    require(asset_records[0]["sha256"] == provenance["sha256"], "WorldAgen checkpoint differs from verified provenance")
    identity = {"protocol": artifact(args.protocol), "prefix_cache": artifact(prefix),
                "registered_initial_adapter": artifact(registered_initial),
                "implementation": [artifact(p) for p in implementation],
                "model_assets": asset_records,
                "contract": CONTRACT, "torch": torch.__version__, "numpy": np.__version__,
                "transformers": importlib.metadata.version("transformers"), "threads": torch.get_num_threads()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "binding.json"
    if manifest.exists():
        require(json.loads(manifest.read_text()) == identity, "Run binding changed; use a new output directory")
    else:
        require(not any(args.out_dir.iterdir()), "Nonempty output without binding")
        core.write_json(manifest, identity)
    cache = torch.load(prefix, map_location="cpu", weights_only=True)
    validate_prefix(cache)
    return protocol, cache, core.file_sha(manifest)


def stage_result(args, stage, binding):
    path = args.out_dir / f"{stage}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    require(result["binding_sha256"] == binding, f"{stage} binding mismatch")
    for value in result.get("artifacts", {}).values():
        check_artifact(value)
    for value in result.get("inputs", {}).values():
        check_artifact(value)
    return result


def completed(args, stage, binding):
    result = stage_result(args, stage, binding)
    require(result is not None and result["status"] == "completed", f"Run completed {stage} first")
    return result


def write_stage(args, stage, binding, result):
    result = {"stage": stage, "binding_sha256": binding, **result}
    core.write_json(args.out_dir / f"{stage}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("stage", "status", "passed", "elapsed_seconds", "completed_windows", "score")}), flush=True)
    return result


def rng_state(device):
    nr = np.random.get_state()
    return {"python": random.getstate(), "numpy_name": nr[0],
            "numpy_keys": torch.tensor(nr[1].astype(np.int64)), "numpy_tail": tuple(nr[2:]),
            "torch_cpu": torch.get_rng_state(),
            "torch_device": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def set_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32), *state["numpy_tail"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_device"], device)


def rng_hash(state):
    h = hashlib.sha256(repr((state["python"], state["numpy_name"], state["numpy_tail"])).encode())
    for key in ("numpy_keys", "torch_cpu", "torch_device"):
        value = state[key]
        if value is not None:
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_model(args, cache, protocol, initial=None):
    # Importing the historical builder does not instantiate any environment.
    # CPU selftests/selection never import it or probe CUDA availability.
    import pilot_worldagen_libero_closed_loop as base
    base.DEVICE = args.device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = base.build_model()
    native, _ = base.masks(model)
    texts = {key: base.clip.tokenize([text], truncate=True).unsqueeze(1).repeat(1, 2, 1).to(args.device)
             for key, text in {**protocol["goals"], "no_lang": "no lang"}.items()}
    before = core.capture_zero_adapter_reference(model, cache, texts["goal0"], native)
    core.seed_all(42, cuda=args.device == "cuda")
    core.attach_lora(model)
    after = core.capture_zero_adapter_reference(model, cache, texts["goal0"], native)
    require(all(torch.equal(before[k], after[k]) for k in before), "Zero adapter changes outputs")
    state = core.snapshot_adapter(model)
    registered = read_tensor(protocol["initial_adapter"])
    require(core.tensor_digest(state) == core.tensor_digest(registered), "Fresh initialization differs from registered original adapter")
    core.restore_adapter(model, registered)
    require(core.tensor_digest(core.snapshot_adapter(model)) == core.tensor_digest(registered), "Restore is not exact")
    require({n for n,p in model.named_parameters() if p.requires_grad} == set(core.adapter_parameters(model)),
            "Unexpected trainable base parameter")
    if initial is not None:
        require(core.tensor_digest(state) == core.tensor_digest(initial), "Initial adapter changed")
    model.attention_mask.copy_(native)
    return model, texts, state


def runtime_metadata(device):
    result = {"device": device, "torch": torch.__version__, "cuda_version": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(),
              "threads": torch.get_num_threads(), "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if device == "cuda":
        result["device_name"] = torch.cuda.get_device_name()
        result["device_capability"] = list(torch.cuda.get_device_capability())
    return result


def peak_memory(device):
    return ({"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
            if device == "cuda" else None)


def bc_loss(model, cache, start, text):
    inputs, _ = core.pack_window(cache, start, text, next(model.parameters()).device)
    arm, gripper, *_ = model(**inputs)
    labels = inputs["action"].reshape(1, 2, 3, 7)
    require(tuple(arm.shape) == (1, 2, 3, 6), f"Policy arm axes differ: {tuple(arm.shape)}")
    require(tuple(gripper.shape) == (1, 2, 3, 1), f"Policy gripper axes differ: {tuple(gripper.shape)}")
    a = F.smooth_l1_loss(arm, labels[..., :6], beta=1.0)
    g = F.binary_cross_entropy(gripper, labels[..., 6:])
    return a + .01 * g, a, g


def norm64(tensors):
    return math.sqrt(math.fsum(float(t.double().square().sum()) for t in tensors.values()))


def gradient_stats(gradients):
    return {"norm": norm64(gradients), "finite": all(bool(torch.isfinite(x).all()) for x in gradients.values()),
            "A_norm": norm64({n: t for n, t in gradients.items() if n.endswith("lora_A")}),
            "B_norm": norm64({n: t for n, t in gradients.items() if n.endswith("lora_B")}),
            "nonzero": sum(int(torch.count_nonzero(t)) for t in gradients.values()),
            "parameters": sum(t.numel() for t in gradients.values())}


def signal(args, protocol, cache, binding):
    prior = stage_result(args, "signal", binding)
    if prior is not None and prior["status"] == "completed":
        return prior
    started = time.monotonic()
    model, texts, initial = load_model(args, cache, protocol)
    model.eval()
    model.zero_grad(set_to_none=True)
    rows = []
    for start in STARTS:
        core.seed_all(20261014 + start, cuda=args.device == "cuda")
        loss, arm, gripper = bc_loss(model, cache, start, texts["goal0"])
        require(bool(torch.isfinite(loss)), "Nonfinite BC surrogate")
        (loss / 30).backward()
        rows.append({"start": start, "loss": float(loss.detach()), "arm": float(arm.detach()),
                     "gripper": float(gripper.detach())})
        print(json.dumps({"stage": "signal", "windows": len(rows), "loss": rows[-1]["loss"]}), flush=True)
    h = core.snapshot_adapter(model, gradients=True)
    stats = gradient_stats(h)
    require(stats["finite"], "Nonfinite BC gradient")
    artifacts = {"initial_adapter": save_tensor(args.out_dir / "initial_adapter.pt", initial),
                 "bc_gradient": save_tensor(args.out_dir / "bc_gradient.pt", h)}
    return write_stage(args, "signal", binding, {"status": "completed", "passed": stats["norm"] > 1e-12,
        "nondegeneracy_threshold": 1e-12, "gradient": stats, "rows": rows,
        "prefix_bc": float(np.mean([r["loss"] for r in rows])), "artifacts": artifacts,
        "device": args.device, "initial_adapter_digest": core.tensor_digest(initial),
        "runtime": runtime_metadata(args.device), "cuda_peak": peak_memory(args.device),
        "token_digest": core.tensor_digest({k: v.cpu() for k, v in texts.items()}),
        "parameter_updates": 0, "environment_steps": 0, "elapsed_seconds": time.monotonic() - started})


def collect(args, protocol, cache, binding):
    sig = completed(args, "signal", binding)
    require(sig["passed"], "Degenerate BC signal; stop before collection")
    require(sig["device"] == args.device, "Do not mix CPU/CUDA gradients")
    prior = stage_result(args, "collect", binding)
    if prior is not None and prior["status"] == "completed":
        return prior
    started = time.monotonic()
    initial = read_tensor(sig["artifacts"]["initial_adapter"])
    model, texts, _ = load_model(args, cache, protocol, initial)
    require(sig["runtime"] == runtime_metadata(args.device), "Model runtime changed after signal")
    model.eval(); model.transformer_backbone.train()
    device = next(model.parameters()).device
    core.seed_all(42, cuda=device.type == "cuda")
    window_dir = args.out_dir / "windows"
    window_dir.mkdir(exist_ok=True)
    rows, artifacts = [], {}
    for index, start in enumerate(STARTS):
        pre = rng_state(device)
        checkpoint = window_dir / f"window_{index:02d}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
            require(row["binding_sha256"] == binding and row["rng_before"] == rng_hash(pre), "Window resume mismatch")
            require(row["start"] == start and row["device"] == args.device, "Window protocol changed")
            for value in row["artifacts"].values(): check_artifact(value)
            post = read_tensor(row["artifacts"]["rng_after"])
            require(rng_hash(post) == row["rng_after"], "Resume RNG artifact differs from recorded hash")
            set_rng(post, device)
        else:
            row = {"binding_sha256": binding, "start": start, "device": args.device,
                   "rng_before": rng_hash(pre), "artifacts": {}, "goals": {}}
            post = None
            for goal in ("goal0", "goal1"):
                model.zero_grad(set_to_none=True)
                set_rng(pre, device)
                inputs, labels = core.pack_window(cache, start, texts[goal], device)
                loss, _, _ = core.forecast_loss(model, inputs, labels)
                require(bool(torch.isfinite(loss)), "Nonfinite world loss")
                (.1 * loss / 30).backward()
                require(all(p.grad is not None for p in core.adapter_parameters(model).values()),
                        "Disconnected LoRA gradient: Adam None semantics require a separate protocol")
                gradients = core.snapshot_adapter(model, gradients=True)
                stats = gradient_stats(gradients)
                require(stats["finite"], "Nonfinite world gradient")
                row["artifacts"][goal] = save_tensor(window_dir / f"window_{index:02d}_{goal}.pt", gradients)
                row["goals"][goal] = {"image_mse": float(loss.detach()), "gradient": stats}
                current_post = rng_state(device)
                if post is not None:
                    require(rng_hash(post) == rng_hash(current_post), "Goal changes RNG consumption")
                post = current_post
                del gradients
            row["artifacts"]["rng_after"] = save_tensor(window_dir / f"window_{index:02d}_rng_after.pt", post)
            row["rng_after"] = rng_hash(post)
            core.write_json(checkpoint, row)
            set_rng(post, device)
        rows.append(row)
        artifacts[f"window_{index:02d}"] = artifact(checkpoint)
        write_stage(args, "collect", binding, {"status": "running", "completed_windows": len(rows),
            "rows": rows, "artifacts": artifacts, "device": args.device,
            "inputs": {"signal": artifact(args.out_dir/"signal.json")},
            "elapsed_seconds": time.monotonic() - started})
    require(core.tensor_digest(core.snapshot_adapter(model)) == core.tensor_digest(initial), "Collection mutated adapter")
    return write_stage(args, "collect", binding, {"status": "completed", "completed_windows": 30,
        "rows": rows, "artifacts": artifacts, "device": args.device, "backward_queries": 60,
        "inputs": {"signal": artifact(args.out_dir/"signal.json")},
        "runtime": runtime_metadata(args.device), "cuda_peak": peak_memory(args.device),
        "parameter_updates": 0, "environment_steps": 0, "elapsed_seconds": time.monotonic() - started})


def accumulate_gradients(gradients, initial, clip=True):
    parameters = [torch.nn.Parameter(t.clone()) for t in initial.values()]
    names = list(initial)
    norms = []
    for row in gradients:
        require(list(row) == names, "Gradient parameter order changed")
        for p, name in zip(parameters, names):
            if p.grad is None: p.grad = row[name].clone()
            else: p.grad.add_(row[name])
        if clip:
            norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, .1, error_if_nonfinite=True,
                                                             foreach=False)))
    return {n: p.grad for n, p in zip(names, parameters)}, norms


def actual_adam_delta(initial, gradient):
    # Actual CPU PyTorch AdamW, no sign-only or hand-derived approximate update.
    parameters = [torch.nn.Parameter(t.clone()) for t in initial.values()]
    opt = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8, foreach=False)
    for p, n in zip(parameters, initial): p.grad = gradient[n].clone()
    opt.step()
    return {n: p.detach() - initial[n] for n, p in zip(initial, parameters)}


def dot64(left, right):
    require(left.keys() == right.keys(), "Parameter keys differ")
    return math.fsum(float((left[n].double() * right[n].double()).sum()) for n in left)


def clipping_certificate(bank):
    total = math.fsum(max(norm64(pair[0]), norm64(pair[1])) for pair in bank)
    # Standard gamma_n bound for sequential FP32 additions. Double-precision
    # norm/sum slack is deliberately larger than their machine precision.
    u = torch.finfo(torch.float32).eps / 2
    gamma30 = 30 * u / (1 - 30 * u)
    # Also allow the worst-case serial FP32 norm reductions (the native norm
    # typically uses shorter parallel reductions). Squares/sqrt and the outer
    # norm over parameter norms receive additional conservative slack.
    n = max(x.numel() for x in bank[0][0].values()) + len(bank[0][0]) + 8
    norm_gamma = n * u / (1 - n * u) if n*u < 1 else math.inf
    upper = math.nextafter(total * (1 + gamma30) * (1 + 4*norm_gamma) * (1 + 1e-10), math.inf)
    # torch clip_grad_norm_ divides by (norm + 1e-6), so norm < max_norm
    # alone does NOT guarantee a unit clip coefficient.
    clip_epsilon, extra_slack = 1e-6, 1e-7
    inactive = upper + clip_epsilon + extra_slack < .1
    return {"triangle_sum": total, "fp32_gamma30": gamma30, "roundoff_upper": upper,
            "fp32_norm_reduction_gamma": norm_gamma,
            "clip_threshold": .1, "clip_denominator_epsilon": clip_epsilon,
            "extra_fp32_slack": extra_slack, "margin": .1 - upper - clip_epsilon - extra_slack,
            "proven_inactive_for_cached_gradient_arithmetic": inactive,
            "scope": "All assignments/all prefixes of cached per-window gradients. Real autograd accumulation is checked separately in verify."}


def initial_assignment():
    return np.random.default_rng(314159).permutation(np.array([0] * 15 + [1] * 15)).tolist()


def balanced_sweep(score, assignment):
    assignment = list(assignment)
    require(assignment.count(0) == assignment.count(1) == 15, "Require15/15")
    current = float(score(assignment)); history = []
    for i in range(30):
        for j in range(i + 1, 30):
            if assignment[i] == assignment[j]:
                history.append({"i": i, "j": j, "skipped_equal_labels": True}); continue
            candidate = list(assignment); candidate[i], candidate[j] = candidate[j], candidate[i]
            value = float(score(candidate)); require(math.isfinite(value), "Nonfinite selection score")
            accepted = value > current
            history.append({"i": i, "j": j, "score_before": current, "candidate_score": value,
                            "accepted": accepted})
            if accepted: assignment, current = candidate, value
    return assignment, current, history


def select(args, protocol, cache, binding):
    sig = completed(args, "signal", binding); col = completed(args, "collect", binding)
    prior = stage_result(args, "select", binding)
    if prior is not None and prior["status"] == "completed": return prior
    started = time.monotonic()
    initial = read_tensor(sig["artifacts"]["initial_adapter"])
    h = read_tensor(sig["artifacts"]["bc_gradient"])
    bank = []
    require(len(col["rows"]) == 30, "Collection incomplete")
    for index, row in enumerate(col["rows"]):
        saved_row = json.loads(check_artifact(col["artifacts"][f"window_{index:02d}"]).read_text())
        require(saved_row == row and row["start"] == STARTS[index], "Collection row changed")
        bank.append([read_tensor(row["artifacts"]["goal0"]), read_tensor(row["artifacts"]["goal1"])])
    cert = clipping_certificate(bank)
    def update(labels):
        g, _ = accumulate_gradients((bank[i][label] for i, label in enumerate(labels)), initial,
                                   clip=not cert["proven_inactive_for_cached_gradient_arithmetic"])
        return actual_adam_delta(initial, g)
    def score(labels):
        return dot64(h, update(labels))
    start = initial_assignment()
    baseline = {"random_balanced": score(start), "uniform_goal0": score([0]*30), "uniform_goal1": score([1]*30)}
    chosen, value, history = balanced_sweep(score, start)
    delta = update(chosen)
    payload = {"assignment": torch.tensor(chosen, dtype=torch.int64), "delta": delta,
               "initial_assignment": torch.tensor(start, dtype=torch.int64)}
    saved = save_tensor(args.out_dir / "selection.pt", payload)
    return write_stage(args, "select", binding, {"status": "completed", "score": value,
        "artifacts": {"selection": saved}, "initial_assignment": start, "assignment": chosen,
        "inputs": {"signal": artifact(args.out_dir/"signal.json"), "collect": artifact(args.out_dir/"collect.json")},
        "baselines": baseline, "history": history, "clipping_certificate": cert,
        "accepted_swaps": sum(r.get("accepted", False) for r in history),
        "strictly_beats_uniform_and_random": value > max(baseline.values()),
        "objective": "BC-gradient dot actual CPU AdamW delta; subtracting any fixed NoLang delta does not alter ordering",
        "not_claimed": ["global optimum", "GPU bit equivalence", "task damage", "forecast mediation"],
        "elapsed_seconds": time.monotonic()-started})


@torch.no_grad()
def prefix_bc(model, cache, text, cuda):
    model.eval(); rows = []
    for start in STARTS:
        core.seed_all(20261014+start, cuda=cuda)
        loss, _, _ = bc_loss(model, cache, start, text)
        rows.append(float(loss))
    return float(np.mean(rows))


def verify(args, protocol, cache, binding):
    sig = completed(args, "signal", binding); col = completed(args, "collect", binding)
    sel = completed(args, "select", binding)
    prior = stage_result(args, "verify", binding)
    if prior is not None and prior["status"] == "completed": return prior
    require(sig["device"] == args.device, "Verification device must match collection")
    started = time.monotonic()
    initial = read_tensor(sig["artifacts"]["initial_adapter"])
    choice = read_tensor(sel["artifacts"]["selection"])
    require(choice["assignment"].tolist() == sel["assignment"], "Selection assignment differs from payload")
    require(choice["initial_assignment"].tolist() == sel["initial_assignment"] == initial_assignment(), "Random control changed")
    require(sel["assignment"].count(0) == sel["assignment"].count(1) == 15, "Selection is not balanced")
    model, texts, _ = load_model(args, cache, protocol, initial)
    require(sig["runtime"] == runtime_metadata(args.device), "Verification runtime changed")
    parameters = list(core.adapter_parameters(model).values())
    arms = {"uniform_goal1": [1]*30, "no_lang": ["no_lang"]*30, "uniform_goal0": [0]*30,
            "random_balanced": sel["initial_assignment"], "selected": sel["assignment"]}
    results, artifacts = {}, {}
    for name, labels in arms.items():
        core.restore_adapter(model, initial)
        model.eval(); model.transformer_backbone.train()
        opt = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=.01)
        opt.zero_grad(set_to_none=True)
        core.seed_all(42, cuda=args.device == "cuda")
        norms = []
        for i, start in enumerate(STARTS):
            key = f"goal{labels[i]}" if isinstance(labels[i], int) else labels[i]
            inputs, targets = core.pack_window(cache, start, texts[key], next(model.parameters()).device)
            loss, _, _ = core.forecast_loss(model, inputs, targets)
            (.1*loss/30).backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, .1, error_if_nonfinite=True)))
        actual_gradient = core.snapshot_adapter(model, gradients=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        adapter = core.snapshot_adapter(model)
        delta = {n: adapter[n]-initial[n] for n in initial}
        actual_bc = prefix_bc(model, cache, texts["goal0"], args.device == "cuda")
        gradient_parity, delta_parity = None, None
        if name != "no_lang":
            cached = [read_tensor(col["rows"][i]["artifacts"][f"goal{label}"]) for i, label in enumerate(labels)]
            expected_g, _ = accumulate_gradients(cached, initial, clip=True)
            expected_delta = actual_adam_delta(initial, expected_g)
            gradient_parity = core.compare_tensors(actual_gradient, expected_g)
            delta_parity = core.compare_tensors(delta, expected_delta)
            del cached, expected_g, expected_delta
        artifacts[name] = save_tensor(args.out_dir/f"verify_{name}_adapter.pt", adapter)
        artifacts[f"{name}_gradient"] = save_tensor(args.out_dir/f"verify_{name}_gradient.pt", actual_gradient)
        results[name] = {"prefix_bc": actual_bc, "delta_bc": actual_bc-sig["prefix_bc"],
            "gradient_parity": gradient_parity, "delta_parity": delta_parity, "clip_norms": norms,
            "actual_linear_score": dot64(read_tensor(sig["artifacts"]["bc_gradient"]), delta),
            "optimizer_updates": 1, "adapter_digest": core.tensor_digest(adapter)}
        print(json.dumps({"stage": "verify", "arm": name, "prefix_bc": actual_bc}), flush=True)
        del actual_gradient, opt
    return write_stage(args, "verify", binding, {"status": "completed", "results": results,
        "artifacts": artifacts, "environment_steps": 0, "task_utility_measured": False,
        "inputs": {"signal": artifact(args.out_dir/"signal.json"), "collect": artifact(args.out_dir/"collect.json"),
                   "select": artifact(args.out_dir/"select.json")},
        "runtime": runtime_metadata(args.device), "cuda_peak": peak_memory(args.device),
        "frozen_prefix_bc": sig["prefix_bc"],
        "uniform_goal1_bridge": "Root must compare the saved adapter/gradient to frozen historical SwapTTT artifacts; this process never opens old task results.",
        "selected_prefix_bc_exceeds_controls": results["selected"]["prefix_bc"] > max(
            results[n]["prefix_bc"] for n in arms if n != "selected"),
        "passed_for_task_gate": False,
        "limitation": "Prefix surrogate only. Inspect numerical parity and preregister the subsequent task/BusCut/ControlCut gate; no automatic task authorization.",
        "elapsed_seconds": time.monotonic()-started})


def self_test():
    import tempfile
    checks = []
    # Explicit forward axes and gripper conversion, no learned model or env.
    cache = {"images": torch.zeros(61, 2, 3, 224, 224), "states": torch.zeros(61, 7),
             "actions": torch.zeros(180, 7), "frame_indices": FRAMES, "train_starts": STARTS}
    cache["actions"][:, -1] = -1
    validate_prefix(cache); checks.append("strict prefix shapes/indices")
    for mutation in ({"heldout_starts": [186]}, {"actions": torch.zeros(181, 7)}):
        try: validate_prefix({**cache, **mutation})
        except ValueError: pass
        else: raise AssertionError("heldout/extra data accepted")
    checks.append("heldout and extra controls rejected")
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.x = torch.nn.Parameter(torch.tensor(.25))
        def forward(self, **kw):
            return self.x.expand(1,2,3,6), self.x.sigmoid().expand(1,2,3,1), None, None, None
    m = Toy(); text = torch.zeros(1,2,77,dtype=torch.long)
    total, arm, grip = bc_loss(m, cache, 0, text)
    assert torch.equal(total, arm+.01*grip)
    assert torch.equal(arm, F.smooth_l1_loss(m.x, torch.tensor(0.)))
    total.backward(); assert m.x.grad != 0
    assert cache["actions"][:, -1].eq(-1).all(); checks.append("official surrogate weights/axes; immutable controls")
    initial = {"a.lora_A": torch.tensor([.4, -.8]), "a.lora_B": torch.zeros(2)}
    micro = [{"a.lora_A": torch.zeros(2), "a.lora_B": torch.tensor([.07, -.2])},
             {"a.lora_A": torch.zeros(2), "a.lora_B": torch.tensor([-.3, .01])}]
    merged, norms = accumulate_gradients(micro, initial)
    direct = [torch.nn.Parameter(t.clone()) for t in initial.values()]
    optimizer = torch.optim.AdamW(direct,lr=5e-4,weight_decay=.01,foreach=False)
    for row in micro:
        sum((p*row[n]).sum() for p,n in zip(direct,initial)).backward()
        torch.nn.utils.clip_grad_norm_(direct,.1,foreach=False)
    optimizer.step()
    expected = {n:p.detach()-initial[n] for n,p in zip(initial,direct)}
    assert all(torch.equal(v,actual_adam_delta(initial,merged)[n]) for n,v in expected.items())
    assert norms[0]>.1 and norms[1]>.1; checks.append("stepwise clipping + actual first AdamW exact CPU")
    state = rng_state(torch.device("cpu")); x = torch.rand(3)
    set_rng(state, torch.device("cpu")); assert torch.equal(x,torch.rand(3)); checks.append("RNG roundtrip")
    labels = initial_assignment(); assert labels.count(0)==labels.count(1)==15
    chosen,value,hist = balanced_sweep(lambda a: sum(i*x for i,x in enumerate(a)),labels)
    assert len(hist)==435 and chosen.count(1)==15 and value>=sum(i*x for i,x in enumerate(labels))
    _,_,ties = balanced_sweep(lambda a: 1.0, labels)
    assert not any(r.get("accepted",False) for r in ties); checks.append("435-pair one sweep balanced, deterministic and strict ties")
    small=[{k:v*1e-4 for k,v in x.items()} for x in micro]
    cert=clipping_certificate([small]*30)
    assert cert["proven_inactive_for_cached_gradient_arithmetic"]
    g1,_=accumulate_gradients([small[0]]*30,initial,True)
    g2,_=accumulate_gradients([small[0]]*30,initial,False)
    assert all(torch.equal(g1[k],g2[k]) for k in g1); checks.append("clipping certificate fast path")
    edge={"a.lora_A":torch.tensor([.0999995/30],dtype=torch.float32)}
    edge_bank=[[edge,edge] for _ in range(30)]
    assert not clipping_certificate(edge_bank)["proven_inactive_for_cached_gradient_arithmetic"]
    edge_init={"a.lora_A":torch.zeros(1)}
    clipped,_=accumulate_gradients([edge]*30,edge_init,True)
    unclip,_=accumulate_gradients([edge]*30,edge_init,False)
    assert not torch.equal(clipped["a.lora_A"],unclip["a.lora_A"])
    checks.append("clip norm+1e-6 near-boundary counterexample rejected")
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"x.pt"; a=save_tensor(p,initial); read_tensor(a)
        p.write_bytes(b"changed")
        try: check_artifact(a)
        except ValueError: pass
        else: raise AssertionError("hash mismatch accepted")
    checks.append("artifact mutation rejected")
    print(json.dumps({"status":"passed","checks":checks,"gpu_work":False,"environment_steps":0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("signal","collect","select","verify"))
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--prefix-cache",type=Path)
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    if args.self_test: self_test(); return
    if args.stage is None or args.protocol is None or args.out_dir is None:
        parser.error("--stage, --protocol and --out-dir are required")
    protocol,cache,binding=prepare(args)
    try:
        globals()[args.stage](args,protocol,cache,binding)
    except Exception as exc:
        core.write_json(args.out_dir/f"{args.stage}_error.json", {
            "stage":args.stage, "status":"error", "binding_sha256":binding,
            "error_type":type(exc).__name__, "error":str(exc),
            "environment_steps":0, "task_failure":False})
        raise


if __name__=="__main__":
    main()
