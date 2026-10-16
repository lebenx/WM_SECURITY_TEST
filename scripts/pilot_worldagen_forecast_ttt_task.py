#!/usr/bin/env python3
"""Forecast-only LoRA adaptation and native LIBERO task evaluation.

This is a reconstruction of the released CALVIN observation-loss/LoRA recipe
for LIBERO, NOT a reproduction of the missing official ttt_libero.py. The
successful local native wrapper supplies the observation/action codec.
All arms replay the same recorded Clean actions. Only the adaptation text or
attention mask changes; deployment always restores native mask/original Goal.

Run --self-test on CPU before inference. The default run first measures Frozen
and CleanTTT on init 0/1/2, and stops before attack evaluation if CleanTTT loses
any Frozen success. Subsequent arms can reuse --data-cache and an accepted
--clean-gate-summary. Existing artifacts are never overwritten or deleted.
"""
from __future__ import annotations

import argparse
from collections import deque
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CONDITIONS = ("Frozen", "CleanTTT", "SwapTTT", "EmptyTTT", "CleanBusCutTTT",
              "SwapBusCutTTT", "SwapControlCutTTT")
SOURCE = ROOT / "runs/worldagen_native_task_gate_confirm_v1/summary.json"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = value.detach().contiguous().cpu()
        digest.update(key.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


class LinearLoRA(nn.Module):
    """W(x) + (alpha/r) B(A(x)); PEFT-default A/zero-B initialization.

    A small explicit implementation avoids requiring PEFT in the successful
    inference runtime. Base weights/bias remain frozen, dropout is exactly 0.
    """

    def __init__(self, base: nn.Linear, rank=64, alpha=32):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                               dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                              dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def attach_lora(model):
    model.requires_grad_(False)
    names = []
    for name, module in list(model.transformer_backbone.named_modules()):
        if name.rsplit(".", 1)[-1] in TARGET_MODULES and isinstance(module, nn.Linear):
            parent_name, leaf = name.rsplit(".", 1)
            parent = model.transformer_backbone.get_submodule(parent_name)
            setattr(parent, leaf, LinearLoRA(module))
            names.append(name)
    expected = len(model.transformer_backbone.layers) * len(TARGET_MODULES)
    if len(names) != expected:
        raise RuntimeError(f"Expected {expected} adapted projections, got {len(names)}")
    return names


def adapter_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}


def snapshot_adapter(model, gradients=False):
    result = {}
    for name, parameter in adapter_parameters(model).items():
        value = parameter.grad if gradients else parameter
        result[name] = torch.zeros_like(parameter, device="cpu") if value is None else value.detach().cpu().clone()
    return result


def restore_adapter(model, state):
    parameters = adapter_parameters(model)
    if parameters.keys() != state.keys():
        raise RuntimeError("Adapter parameter mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])
            parameter.grad = None


def compare_tensors(left, right):
    if left.keys() != right.keys():
        raise RuntimeError("Tensor comparison keys differ")
    maximum, squared, reference, exact = 0.0, 0.0, 0.0, True
    for key in left:
        x, y = left[key].double(), right[key].double()
        delta = x - y
        maximum = max(maximum, float(delta.abs().max()))
        squared += float(delta.square().sum())
        reference += float(x.square().sum())
        exact = exact and torch.equal(left[key], right[key])
    return {"exact": exact, "max_abs": maximum, "l2": math.sqrt(squared),
            "relative_l2": math.sqrt(squared / max(reference, 1e-30))}


def patchify(images, patch_size=16):
    """Exact released patch layout on CLIP-normalized RGB; NO patch whitening."""
    n, c, h, w = images.shape
    if c != 3 or h != w or h % patch_size:
        raise ValueError("Expected square RGB images divisible by patch size")
    x = images.reshape(n, 3, h // patch_size, patch_size, w // patch_size, patch_size)
    return torch.einsum("nchpwq->nhwpqc", x).reshape(n, (h // patch_size) ** 2, patch_size ** 2 * 3)


def window_starts(num_actions, train_windows=30, heldout_windows=20):
    train = list(range(0, train_windows * 6, 6))
    remaining = list(range(train_windows * 6, num_actions - 5, 6))
    if len(remaining) < heldout_windows:
        raise ValueError(f"Need {train_windows + heldout_windows} disjoint six-action windows; "
                         f"trace has only {num_actions} actions")
    selected = np.linspace(0, len(remaining) - 1, heldout_windows, dtype=int)
    heldout = [remaining[int(i)] for i in selected]
    if set(train) & set(heldout):
        raise AssertionError("Training/heldout start overlap")
    return train, heldout


def validate_source(args):
    source = json.loads(args.source_summary.read_text())
    protocol = source["protocol"]
    if source["status"] != "completed":
        raise ValueError("Source Clean native task run is not complete")
    for key, expected in {"suite": args.suite, "seed": args.seed, "camera_size": args.camera_size,
                          "settle_steps": args.settle_steps, "orientation": args.orientation}.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Source protocol {key}={protocol.get(key)!r}, requested {expected!r}")
    rows = [row for row in source["results"] if row["task_id"] == args.task_id
            and row["init_id"] == args.source_init_id and row["condition"] == "native"]
    if len(rows) != 1 or not rows[0]["success"] or rows[0]["success_at_start"]:
        raise ValueError("Require exactly one successful nontrivial native source episode")
    row = rows[0]
    trace = Path(row["trace"])
    records = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if [r["step"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Non-contiguous source action trace")
    if len(records) != row["environment_steps"]:
        raise ValueError("Source trace length differs from episode record")
    return row, records, {"summary_path": str(args.source_summary.resolve()),
                          "summary_sha256": file_sha(args.source_summary),
                          "trace_path": str(trace.resolve()), "trace_sha256": file_sha(trace)}


def replay_data(model, task, source_row, records, provenance, args, gate):
    """Replay actual Clean controls and verify every available original observable.

    The original trace has no pixel/complete-physics snapshots. We record that
    limitation explicitly instead of claiming an exact full-state replay.
    """
    base = gate.base
    train, heldout = window_starts(len(records), args.train_windows, args.heldout_windows)
    needed = sorted({start + offset for start in train + heldout for offset in (0, 3, 6)})
    path = base.LIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = base.LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    if file_sha(path) != source_row["bddl_sha256"]:
        raise RuntimeError("Source BDDL changed")
    initial = torch.load(init_path, map_location="cpu", weights_only=False)[args.source_init_id]
    if hashlib.sha256(np.asarray(initial).tobytes()).hexdigest() != source_row["initial_state_sha256"]:
        raise RuntimeError("Source initial state changed")
    seed_all(args.seed, cuda=args.device == "cuda")
    env = base.OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=args.camera_size,
                                camera_widths=args.camera_size, render_gpu_device_id=0)
    images, states, observed_checks = [], [], []
    maximum = {"eef_before": 0.0, "eef_after": 0.0, "eef_quaternion": 0.0, "gripper_qpos": 0.0}
    last_gripper = -1.0

    def keep(obs):
        images.append(torch.stack([model.image_processor(base.Image.fromarray(obs[key]).convert("RGB"))
                                   for key in ("agentview_image", "robot0_eye_in_hand_image")]))
        states.append(gate.prepare_state(obs, last_gripper, args.orientation)[0, 0])

    try:
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(args.settle_steps):
            obs, *_ = env.step(np.zeros(7, dtype=np.float32))
        if env.check_success():
            raise RuntimeError("Replay task already successful after settling")
        keep(obs)
        for index, record in enumerate(records):
            if index >= max(needed):
                break
            # The native wrapper casts controls to CPU fp16 before env.step.
            # Even exactly representable fp16 values dispatched as fp32 can
            # change simulator/controller arithmetic, so preserve the dtype.
            action = np.asarray(record["action"], dtype=np.float16)
            before = np.asarray(obs["robot0_eef_pos"])
            obs, reward, done, _ = env.step(action)
            last_gripper = float(action[-1])
            actual = {"eef_before": before, "eef_after": obs["robot0_eef_pos"],
                      "eef_quaternion": obs["robot0_eef_quat"], "gripper_qpos": obs["robot0_gripper_qpos"]}
            for key, value in actual.items():
                error = float(np.max(np.abs(np.asarray(value) - record[key])))
                maximum[key] = max(maximum[key], error)
                if error > args.replay_atol:
                    raise RuntimeError(f"Source replay mismatch at step {index+1}, {key}: {error}")
            if float(reward) != record["reward"] or bool(done) != record["done"] or bool(env.check_success()) != record["task_success"]:
                raise RuntimeError(f"Source replay reward/success mismatch at step {index+1}")
            if index + 1 in needed:
                keep(obs)
            observed_checks.append(index + 1)
        if len(images) != len(needed):
            raise RuntimeError("Replay omitted required observations")
    finally:
        env.close()
    return {"schema": 1, "source": provenance, "task_name": task.name, "clean_goal": task.language,
            "frame_indices": needed, "images": torch.stack(images), "states": torch.stack(states),
            "actions": torch.tensor([row["action"] for row in records], dtype=torch.float32),
            "train_starts": train, "heldout_starts": heldout,
            "replay": {"checked_steps": len(observed_checks), "max_abs": maximum,
                       "atol": args.replay_atol, "reward_done_success_exact": True,
                       "env_action_dtype": "float16 (native wrapper output)",
                       "source_pixels_and_full_physics_available": False},
            "preprocessing": "Raw native env RGB -> model.image_processor (CLIP); no rotation, no per-patch whitening",
            "data_split": "30 disjoint six-action training windows; later same-trajectory windows held out. "
                          "Adjacent windows share boundary observation; not independent-trajectory validation."}


def pack_window(cache, start, text, device):
    lookup = {value: index for index, value in enumerate(cache["frame_indices"])}
    source = [lookup[start], lookup[start + 3]]
    target = [lookup[start + 3], lookup[start + 6]]
    views = cache["images"][source].to(device)
    future = cache["images"][target].to(device)
    # Native image_pred axes are [batch*time, view, patch, RGB_patch_values].
    labels = patchify(future.flatten(0, 1)).reshape(2, 2, 196, 768)
    state = cache["states"][source].unsqueeze(0).to(device)
    actions = cache["actions"][start:start + 6].unsqueeze(0).to(device).clone()
    # Native training maps binary commands {-1,+1} to {0,1}; forward's
    # binary one-hot maps either encoding identically. This is NOT a policy label.
    actions[..., 6] = (actions[..., 6] + 1) / 2
    state = state.clone()
    state[..., 6] = (state[..., 6] + 1) / 2
    inputs = {"image_primary": views[:, 0].unsqueeze(0), "image_wrist": views[:, 1].unsqueeze(0),
              "state": state, "text_token": text, "action": actions}
    return inputs, labels


def forecast_loss(model, inputs, labels):
    _, _, prediction, _, _ = model(**inputs)
    if prediction.shape != labels.shape:
        raise RuntimeError(f"Forecast/target axes differ: {prediction.shape} != {labels.shape}")
    primary = F.mse_loss(prediction[:, 0], labels[:, 0])
    wrist = F.mse_loss(prediction[:, 1], labels[:, 1])
    return .5 * (primary + wrist), primary, wrist


@torch.no_grad()
def capture_zero_adapter_reference(model, cache, text, mask):
    """Compare real policy/world outputs before/after attaching zero-B LoRA."""
    model.eval()
    model.attention_mask.copy_(mask)
    device = next(model.parameters()).device
    start = cache["train_starts"][0]
    seed_all(20261014 + int(start), cuda=device.type == "cuda")
    inputs, _ = pack_window(cache, start, text, device)
    arm, gripper, world, _, _ = model(**inputs)
    return {"policy_arm": arm.detach().cpu().clone(),
            "policy_gripper": gripper.detach().cpu().clone(),
            "world": world.detach().cpu().clone()}


@torch.no_grad()
def evaluate_forecasts(model, cache, starts, text, mask):
    model.eval()
    model.attention_mask.copy_(mask)
    device = next(model.parameters()).device
    rows = []
    for start in starts:
        # MAE permutes patches even at mask_ratio=0. Reuse identical per-window
        # randomness across before/after evaluation and every adaptation arm.
        seed_all(20261014 + int(start), cuda=device.type == "cuda")
        inputs, labels = pack_window(cache, start, text, device)
        loss, primary, wrist = forecast_loss(model, inputs, labels)
        rows.append({"start": start, "image_mse": float(loss),
                     "primary_mse": float(primary), "wrist_mse": float(wrist)})
    return {"windows": len(rows), "image_mse_mean": float(np.mean([r["image_mse"] for r in rows])), "rows": rows}


def train_one_update(model, cache, text, mask, args):
    model.eval()  # Frozen image/CLIP encoders and decoder stay deterministic.
    model.transformer_backbone.train()
    model.attention_mask.copy_(mask)
    parameters = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    seed_all(args.adapter_seed, cuda=args.device == "cuda")
    rows = []
    for index, start in enumerate(cache["train_starts"]):
        inputs, labels = pack_window(cache, start, text, next(model.parameters()).device)
        loss, primary, wrist = forecast_loss(model, inputs, labels)
        scaled = .1 * loss / len(cache["train_starts"])
        if not bool(torch.isfinite(scaled)):
            raise RuntimeError("Nonfinite TTT loss")
        scaled.backward()
        # Match ttt.py exactly: clip after EACH microbatch, not just the update.
        norm = torch.nn.utils.clip_grad_norm_(parameters, .1, error_if_nonfinite=True)
        rows.append({"start": start, "image_mse": float(loss.detach()),
                     "primary_mse": float(primary.detach()), "wrist_mse": float(wrist.detach()),
                     "scaled_loss": float(scaled.detach()), "pre_clip_accumulated_grad_norm": float(norm)})
        if (index + 1) % 5 == 0:
            print(json.dumps({"ttt_microbatches": index + 1, "optimizer_updates": 0,
                              "image_mse": rows[-1]["image_mse"]}), flush=True)
    gradients = snapshot_adapter(model, gradients=True)
    optimizer.step()  # Exactly ONE update; official cosine scheduler follows it.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    return {"optimizer_updates": 1, "microbatches": len(rows), "rows": rows,
            "gradient_sha256": tensor_digest(gradients)}, gradients


def make_masks(model, gate):
    native, bus = gate.base.masks(model)
    control = native.clone()
    control[[37, 38, 39, 77, 78, 79, 77, 78, 79], [0, 0, 0, 0, 0, 0, 40, 40, 40]] = -float("inf")
    for name, mask in (("bus", bus), ("control", control)):
        if int((torch.isfinite(native) & ~torch.isfinite(mask)).sum()) != 9:
            raise RuntimeError(f"{name} mask does not remove exactly nine finite edges")
    return native, bus, control


def install_device_wrapper(gate, device):
    """Override only the native wrapper initializer's hard-coded CUDA device.

    The action/world/reset methods remain the upstream implementations. A CPU
    run has its own Frozen baseline and protocol identity and cannot reuse a
    CUDA clean gate. The original sampler already returns fp16 CPU controls.
    """
    original = gate.base.WorldAgenLiberoModelWrapper

    class DeviceNativeWrapper(original):
        def __init__(self, args, model, tokenizer, image_processor, cast_dtype,
                     history_len=10, libero_eval_max_steps=600):
            self.device = device
            self.model = model.to(device)
            self.cast_type = cast_dtype
            self.text_process_fn = functools.partial(gate.base.preprocess_text_calvin, tokenizer=tokenizer)
            self.image_process_fn = functools.partial(gate.base.preprocess_image, image_processor=image_processor)
            self.action_hist_queue = []
            self.history_len = history_len
            self.libero_eval_max_steps = libero_eval_max_steps
            self.img_queue = deque(maxlen=history_len)
            self.gripper_queue = deque(maxlen=history_len)
            self.state_queue = deque(maxlen=history_len)
            self.mask_queue = deque(maxlen=history_len)
            self.text_queue = deque(maxlen=history_len)
            self.action_queue = deque(maxlen=history_len)
            self.args = args
            self.gripper_state = np.array([-1.0])
            self.cnt = 0

    gate.base.DEVICE = device
    gate.base.WorldAgenLiberoModelWrapper = DeviceNativeWrapper


def check_clean_gate(results):
    if not all(name in results for name in ("Frozen", "CleanTTT")):
        return {"status": "not_yet_evaluated", "passed": False}
    frozen = {row["init_id"]: row for row in results["Frozen"]["episodes"]}
    adapted = {row["init_id"]: row for row in results["CleanTTT"]["episodes"]}
    if frozen.keys() != adapted.keys():
        raise RuntimeError("Clean gate episode keys differ")
    successes = sum(row["success"] for row in frozen.values())
    lost = [key for key in frozen if frozen[key]["success"] and not adapted[key]["success"]]
    return {"status": "evaluated", "passed": successes > 0 and not lost,
            "frozen_successes": successes, "clean_ttt_successes": sum(row["success"] for row in adapted.values()),
            "episodes": len(frozen), "lost_frozen_success_init_ids": lost,
            "criterion": "At least one Frozen success and no Frozen-success episode lost after CleanTTT; "
                         "development gate, not statistical equivalence."}


def compare_artifacts(results, left, right, output):
    if left not in results or right not in results:
        return None
    x = torch.load(output / results[left]["adapter_file"], map_location="cpu", weights_only=True)
    y = torch.load(output / results[right]["adapter_file"], map_location="cpu", weights_only=True)
    gx = torch.load(output / results[left]["gradient_file"], map_location="cpu", weights_only=True)
    gy = torch.load(output / results[right]["gradient_file"], map_location="cpu", weights_only=True)
    losses_x = torch.tensor([r["image_mse"] for r in results[left]["training"]["rows"]], dtype=torch.float64)
    losses_y = torch.tensor([r["image_mse"] for r in results[right]["training"]["rows"]], dtype=torch.float64)
    return {"adapters": compare_tensors(x, y), "gradients": compare_tensors(gx, gy),
            "train_losses": compare_tensors({"losses": losses_x}, {"losses": losses_y}),
            "episodes_identical_actions": [file_sha(a["trace"]) == file_sha(b["trace"])
                                            for a, b in zip(results[left]["episodes"], results[right]["episodes"])]}


def self_test():
    """CPU-only checks of loss layout, adapter identity and accumulation semantics."""
    seed_all(17)
    base = nn.Linear(5, 4)
    original = {k: v.detach().clone() for k, v in base.named_parameters()}
    layer = LinearLoRA(base, rank=3, alpha=2)
    x, target = torch.randn(2, 5), torch.randn(2, 4)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    F.mse_loss(layer(x), target).backward()
    assert torch.count_nonzero(layer.lora_A.grad) == 0
    assert torch.count_nonzero(layer.lora_B.grad) > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    with torch.no_grad():
        layer.lora_B.normal_()
    explicit = base(x) + (x @ layer.lora_A.T @ layer.lora_B.T) * (2 / 3)
    torch.testing.assert_close(layer(x), explicit)
    assert all(torch.equal(value, original[key]) for key, value in base.named_parameters())
    values = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    patched = patchify(values)
    expected = values[0, :, :16, :16].permute(1, 2, 0).reshape(-1)
    torch.testing.assert_close(patched[0, 0], expected, rtol=0, atol=0)
    assert float(patched.mean()) > 100  # Explicitly reject accidental patch whitening.
    train, heldout = window_starts(321)
    assert len(train) == 30 and len(heldout) == 20
    train_actions = {a for start in train for a in range(start, start + 6)}
    heldout_actions = {a for start in heldout for a in range(start, start + 6)}
    assert not train_actions.intersection(heldout_actions)
    assert set(CONDITIONS) == {"Frozen", "CleanTTT", "SwapTTT", "EmptyTTT", "CleanBusCutTTT",
                               "SwapBusCutTTT", "SwapControlCutTTT"}
    # A uniform target offset stays visible in the true RGB-patch error.
    assert F.mse_loss(patched, patched + 3).item() == 9
    cache = {"frame_indices": [0, 3, 6],
             "images": torch.stack([torch.full((2, 3, 224, 224), float(i)) for i in (0, 3, 6)]),
             "states": torch.zeros(3, 7), "actions": torch.zeros(6, 7)}
    cache["states"][:, -1] = -1
    cache["actions"][:, -1] = -1
    cache["actions"][:, 0] = torch.arange(6)
    inputs, labels = pack_window(cache, 0, torch.zeros(1, 2, 77, dtype=torch.int64), "cpu")
    assert inputs["image_primary"][0, 0].mean() == 0
    assert inputs["image_primary"][0, 1].mean() == 3
    assert labels.shape == (2, 2, 196, 768)
    assert labels[0].mean() == 3 and labels[1].mean() == 6
    assert torch.equal(inputs["action"][0, :, 0], torch.arange(6).float())
    assert inputs["action"][..., -1].sum() == 0
    assert cache["actions"][..., -1].sum() == -6  # Cache must not be modified.
    assert tensor_digest({"scalar": torch.tensor(1.)}) == tensor_digest({"scalar": torch.tensor(1.)})
    tiny = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=5e-4, weight_decay=.01)
    before = tiny.weight.detach().clone()
    for _ in range(30):
        (.1 * tiny(torch.ones(1, 2)).square().mean() / 30).backward()
        torch.nn.utils.clip_grad_norm_(tiny.parameters(), .1)
        assert torch.equal(tiny.weight, before)
    optimizer.step()
    assert int(optimizer.state[tiny.weight]["step"]) == 1
    print(json.dumps({"status": "passed", "device": "cpu", "gpu_inference": False,
                      "checks": ["zero-B identity", "LoRA equation", "frozen base gradients/weights",
                                 "RGB patch layout/no whitening", "0,3->3,6 teacher alignment",
                                 "disjoint action split", "30 microbatches/1 update", "scalar tensor hashing"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true",
                        help="Replay fixed source controls and cache CPU-processed RGB; no model load/forward, LoRA or episodes")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-summary", type=Path, default=SOURCE)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--clean-gate-summary", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--source-init-id", type=int, default=0)
    parser.add_argument("--eval-init-ids", default="0,1,2")
    parser.add_argument("--swap-task-id", type=int)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--adapter-seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--orientation", choices=("euler", "train_axis_angle"), default="euler")
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--log-every-chunks", type=int, default=25)
    parser.add_argument("--train-windows", type=int, default=30)
    parser.add_argument("--heldout-windows", type=int, default=20)
    parser.add_argument("--replay-atol", type=float, default=1e-9)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for experiments")
    if args.train_windows != 30 or args.heldout_windows < 1:
        parser.error("This registered single-update pilot requires exactly 30 training windows")
    if args.camera_size != 224 or args.max_steps < 1 or args.replay_atol <= 0 or args.threads < 1:
        parser.error("Require 224-pixel camera and positive evaluation/replay limits")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Duplicate conditions")
    args.eval_init_ids = [int(item) for item in args.eval_init_ids.split(",")]
    if len(set(args.eval_init_ids)) != len(args.eval_init_ids) or not args.eval_init_ids or min(args.eval_init_ids) < 0:
        parser.error("Invalid evaluation initial-state IDs")
    if (not args.prepare_data_only and any(c not in ("Frozen", "CleanTTT") for c in args.conditions)
            and not args.clean_gate_summary):
        if args.conditions[:2] != ["Frozen", "CleanTTT"]:
            parser.error("Run Frozen then CleanTTT first, or supply a matching passed --clean-gate-summary")
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; existing evidence will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    import pilot_worldagen_native_task_gate as gate
    if not args.prepare_data_only and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu with a separate Frozen baseline")
    torch.set_num_threads(args.threads)
    install_device_wrapper(gate, args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_row, records, source_provenance = validate_source(args)
    suite = gate.base.benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    swap_id = (args.task_id + 1) % suite.n_tasks if args.swap_task_id is None else args.swap_task_id
    if swap_id == args.task_id or not 0 <= swap_id < suite.n_tasks:
        parser.error("Swap must be a different legal task label from the same suite")
    swap_goal = suite.get_task(swap_id).language
    protocol = {"description": __doc__, "script_sha256": file_sha(__file__), "source": source_provenance,
                "suite": args.suite, "task_id": args.task_id, "source_init_id": args.source_init_id,
                "eval_init_ids": args.eval_init_ids, "swap_task_id": swap_id,
                "clean_goal": task.language, "swap_goal": swap_goal,
                "environment_seed": args.seed, "adapter_seed": args.adapter_seed,
                "model_device": args.device, "torch_threads": args.threads,
                "cpu_scope": "Separate CPU Frozen/TTT baseline; CUDA source actions used only for fixed-data replay. "
                             "CPU and CUDA success/gradient results must not be pooled as paired conditions.",
                "max_steps": args.max_steps, "camera_size": args.camera_size,
                "orientation": args.orientation, "settle_steps": args.settle_steps,
                "train_windows": args.train_windows, "heldout_windows": args.heldout_windows,
                "lora": {"rank": 64, "alpha": 32, "dropout": 0, "projections": TARGET_MODULES,
                         "implementation": "explicit torch LinearLoRA; same A/B equation/init as PEFT, not PEFT execution"},
                "optimizer": {"type": "AdamW", "lr": 5e-4, "weight_decay": .01,
                              "loss_scale": .1, "gradient_accumulation": 30, "updates": 1,
                              "clip_norm": .1, "clip_after_every_microbatch": True,
                              "schedule": "single non-warmup update at configured lr; no later scheduler-dependent update"},
                "loss": "0.1 * mean(primary/wrist MSE), CLIP RGB patches, no per-patch whitening, no policy loss",
                "adaptation_mode": "Qwen backbone train; frozen vision/CLIP/decoder eval; LoRA dropout zero",
                "forecast_evaluation_rng": "seed 20261014 + window start, reset for every pre/post/arm query",
                "deployment": "Original task Goal and native mask for every condition; no swap/cut persists at inference",
                "scope": "Reconstructed forecast-only LIBERO LoRA adapter; official ttt_libero.py absent. "
                         "Fixed Clean-trajectory collection replaces exploratory sampling. Source-init evaluation is "
                         "a development test; other evaluation init IDs are independent of adaptation trajectory.",
                "not_claimed": ["exact official LIBERO TTT reproduction", "native external attacker access", "forecast consumed by frozen policy"],
                "checkpoint_sha256": json.loads((gate.base.MODEL_DIR / "provenance.json").read_text())["sha256"],
                "source_files": {str(path.relative_to(ROOT)): file_sha(path) for path in [
                    Path(gate.__file__), Path(gate.base.__file__),
                    ROOT / "vendor/WorldAgen/test_time_training/ttt.py",
                    ROOT / "vendor/WorldAgen/test_time_training/lora.py",
                    ROOT / "vendor/WorldAgen/scripts/libero/eval_ttt.sh"]}}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    results = {}
    summary = {"status": "preparing", "protocol": protocol,
               "conditions": [] if args.prepare_data_only else args.conditions,
               "prepare_data_only": args.prepare_data_only,
               "api_requests": 0, "results": results, "clean_gate": {"passed": False, "status": "not_yet_evaluated"}}
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    model = None
    frozen_digest = None
    try:
        # clip.load returns this exact transform for the released ViT-B/32
        # (224 resolution). Data preparation needs no encoder/checkpoint load.
        from clip.clip import _transform
        processor = SimpleNamespace(image_processor=_transform(224))
        cache_path = (args.data_cache or (output / "data.pt")).resolve()
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["source"] != source_provenance or cache["clean_goal"] != task.language:
                raise RuntimeError("Cache source/Goal differs from requested source")
            expected_train, expected_heldout = window_starts(len(records), args.train_windows, args.heldout_windows)
            if cache["train_starts"] != expected_train or cache["heldout_starts"] != expected_heldout:
                raise RuntimeError("Cache train/heldout window split differs")
        else:
            cache = replay_data(processor, task, source_row, records, source_provenance, args, gate)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("xb") as stream:
                torch.save(cache, stream)
        summary["data"] = {"path": str(cache_path), "sha256": file_sha(cache_path),
                           "replay": cache["replay"], "train_starts": cache["train_starts"],
                           "heldout_starts": cache["heldout_starts"], "preprocessing": cache["preprocessing"]}
        if args.prepare_data_only:
            summary["status"] = "prepared_data_only"
            summary["model_loaded"] = False
            summary["model_forward_calls"] = 0
            summary["optimizer_updates"] = 0
            summary["task_evaluation_episodes"] = 0
            summary["scientific_scope"] = "Verified fixed-control replay and CPU preprocessing only; no task success-rate or attack/defense result."
            write_json(summary_path, summary)
            print(json.dumps({"status": summary["status"], "data": summary["data"]}), flush=True)
            return
        # A staged arm must use the very same implementation as its Clean
        # gate; a later code change requires revalidating that gate.
        identity = {key: value for key, value in protocol.items() if key != "description"}
        identity["cache_sha256"] = summary["data"]["sha256"]
        summary["protocol_identity_sha256"] = json_sha(identity)
        if args.clean_gate_summary:
            prior = json.loads(args.clean_gate_summary.read_text())
            if not prior["clean_gate"]["passed"] or prior.get("protocol_identity_sha256") != summary["protocol_identity_sha256"]:
                raise RuntimeError("Prior clean gate did not pass or protocol/data identity differs")
            summary["clean_gate"] = prior["clean_gate"]
            summary["clean_gate_source"] = {"path": str(args.clean_gate_summary.resolve()), "sha256": file_sha(args.clean_gate_summary)}
        model = gate.base.build_model()
        native, bus, control = make_masks(model, gate)
        texts = {name: gate.base.clip.tokenize([value], truncate=True).unsqueeze(1).repeat(1, 2, 1).to(gate.base.DEVICE)
                 for name, value in (("clean", task.language), ("swap", swap_goal), ("empty", ""))}
        before_attach = capture_zero_adapter_reference(model, cache, texts["clean"], native)
        seed_all(args.adapter_seed, cuda=args.device == "cuda")
        adapted_names = attach_lora(model)
        after_attach = capture_zero_adapter_reference(model, cache, texts["clean"], native)
        summary["zero_adapter_equivalence"] = {"model_device": args.device,
                                               "window_start": cache["train_starts"][0],
                                               "seed": 20261014 + cache["train_starts"][0],
                                               "outputs": {key: compare_tensors({key: before_attach[key]}, {key: after_attach[key]})
                                                           for key in before_attach}}
        write_json(summary_path, summary)
        if not all(row["exact"] for row in summary["zero_adapter_equivalence"]["outputs"].values()):
            raise RuntimeError("Real cached-window outputs changed after attaching zero-B LoRA")
        del before_attach, after_attach
        initial_adapter = snapshot_adapter(model)
        torch.save(initial_adapter, output / "initial_adapter.pt")
        summary["lora_projection_count"] = len(adapted_names)
        summary["trainable_parameter_count"] = sum(p.numel() for p in adapter_parameters(model).values())
        summary["initial_adapter_sha256"] = tensor_digest(initial_adapter)
        adapter_names = set(adapter_parameters(model))
        frozen_digest = tensor_digest({n: p for n, p in model.named_parameters() if n not in adapter_names})
        summary["frozen_parameter_sha256_before"] = frozen_digest
        arms = {"Frozen": ("clean", native), "CleanTTT": ("clean", native),
                "SwapTTT": ("swap", native), "EmptyTTT": ("empty", native),
                "CleanBusCutTTT": ("clean", bus), "SwapBusCutTTT": ("swap", bus),
                "SwapControlCutTTT": ("swap", control)}
        for condition in args.conditions:
            if condition not in ("Frozen", "CleanTTT") and not summary["clean_gate"]["passed"]:
                summary["status"] = "stopped_clean_ttt_gate_failed"
                summary["remaining_conditions_not_run"] = [c for c in args.conditions if c not in results]
                break
            print(json.dumps({"starting": condition}), flush=True)
            summary["status"] = "running"
            summary["current_condition"] = condition
            write_json(summary_path, summary)
            restore_adapter(model, initial_adapter)
            text_name, mask = arms[condition]
            seed_all(args.adapter_seed, cuda=args.device == "cuda")
            pre = evaluate_forecasts(model, cache, cache["heldout_starts"], texts[text_name], mask)
            if condition != "Frozen":
                training, gradients = train_one_update(model, cache, texts[text_name], mask, args)
            else:
                training, gradients = {"optimizer_updates": 0, "microbatches": 0, "rows": []}, {}
            adapter = snapshot_adapter(model)
            torch.save(adapter, output / f"{condition}_adapter.pt")
            torch.save(gradients, output / f"{condition}_gradients.pt")
            result = {"condition": condition, "adaptation_text": text_name,
                      "training": training, "heldout_before_adaptation_condition": pre,
                      "heldout_after_adaptation_condition": evaluate_forecasts(model, cache, cache["heldout_starts"], texts[text_name], mask),
                      "heldout_after_clean_native": evaluate_forecasts(model, cache, cache["heldout_starts"], texts["clean"], native),
                      "adapter_file": f"{condition}_adapter.pt", "gradient_file": f"{condition}_gradients.pt",
                      "adapter_sha256": tensor_digest(adapter),
                      "adapter_change": compare_tensors(initial_adapter, adapter), "episodes": []}
            del adapter, gradients
            results[condition] = result
            write_json(summary_path, summary)
            model.eval()
            model.attention_mask.copy_(native)
            for init_id in args.eval_init_ids:
                # episode sets native mask and task.language again; no attack text reaches policy.
                episode = gate.episode(model, native, task, args.suite, args.task_id,
                                       init_id, condition, args, output)
                episode["model_device"] = args.device
                result["episodes"].append(episode)
                if condition == "Frozen" and init_id == args.source_init_id:
                    exact = file_sha(episode["trace"]) == source_provenance["trace_sha256"]
                    result["source_native_trace_exact"] = exact
                    result["source_native_trace_comparison_device_matched"] = args.device == "cuda"
                    if not exact and args.device == "cuda":
                        raise RuntimeError("Zero-initialized LoRA Frozen episode does not exactly reproduce the source native trace")
                write_json(summary_path, summary)
            result["task_success_rate"] = float(np.mean([r["success"] for r in result["episodes"]]))
            result["task_successes"] = sum(r["success"] for r in result["episodes"])
            if "Frozen" in results and "CleanTTT" in results:
                summary["clean_gate"] = check_clean_gate(results)
            summary["elapsed_seconds"] = time.monotonic() - started
            write_json(summary_path, summary)
        else:
            summary["status"] = "completed"
        summary["causal_comparisons"] = {
            "clean_vs_swap_cut": compare_artifacts(results, "CleanBusCutTTT", "SwapBusCutTTT", output),
            "swap_vs_matched_control": compare_artifacts(results, "SwapTTT", "SwapControlCutTTT", output)}
        summary["aggregate"] = {name: {"success_rate": row["task_success_rate"], "successes": row["task_successes"],
                                      "episodes": len(row["episodes"])} for name, row in results.items()}
        summary["scientific_scope"] = (
            "Development evidence only; no attack/repair inference unless actual success losses/recovery occur. "
            "EmptyTTT is the ordinary text-canonicalization defense baseline.")
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if model is not None and frozen_digest is not None:
            adapter_names = set(adapter_parameters(model))
            after = tensor_digest({n: p for n, p in model.named_parameters() if n not in adapter_names})
            summary["frozen_parameter_sha256_after"] = after
            summary["frozen_parameters_unchanged"] = after == frozen_digest
            if after != frozen_digest:
                summary["status"] = "invalid_frozen_weights_changed"
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "status": summary["status"],
                      "clean_gate": summary["clean_gate"], "aggregate": summary.get("aggregate")}), flush=True)


if __name__ == "__main__":
    main()
