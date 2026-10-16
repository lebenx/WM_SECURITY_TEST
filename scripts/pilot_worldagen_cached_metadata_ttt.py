#!/usr/bin/env python3
"""Cached-text poisoning diagnostic for reconstructed WorldAgen LIBERO TTT.

The released CALVIN sampler writes literal ``no lang`` into its adaptation
cache. A legal policy Goal cannot override that canonicalization. This runner
assumes an attacker can instead rewrite ONLY pretokenized cached ``text``;
physical observations, executed actions, target images, training RNG, and the
deployed policy Goal remain fixed. Cache-writer access is an explicit threat
assumption, not an established native external attack surface.

NoLangTTT transplants the CALVIN no-lang convention to LIBERO. The official
LIBERO trainer is absent, so this is not its reproduction. GoalTTT (formerly
CleanTTT) is only an original-goal metadata diagnostic. Frozen + NoLangTTT is
the default health check. Other arms require that health gate AND a completed
matched 30-episode native clean gate (21 successes, at least 7 tasks).

Shared numerical routines come from pilot_worldagen_forecast_ttt_task.py;
this entry point does not modify that script or the native policy runner.
Physical cache schema 3 purges one six-action window between training and
heldout windows, so no image index crosses the partition. Older caches are
rejected. Windows still belong to one trajectory and are not independent
episodes.
--self-test uses CPU only. --prepare-data-only performs replay without model
forward. No inference or experiment is launched merely by importing it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import torch

import pilot_worldagen_forecast_ttt_task as core
from pilot_worldagen_forecast_ttt_task import (
    ROOT, SOURCE, TARGET_MODULES, adapter_parameters, attach_lora,
    capture_zero_adapter_reference, compare_artifacts, compare_tensors,
    evaluate_forecasts, file_sha, install_device_wrapper, json_sha, make_masks,
    restore_adapter, seed_all, snapshot_adapter, tensor_digest,
    train_one_update, write_json,
)

CONDITIONS = ("Frozen", "NoLangTTT", "GoalTTT", "SwapTTT", "EmptyTTT",
              "NoLangBusCutTTT", "SwapBusCutTTT", "SwapControlCutTTT")
HEALTH_CONDITIONS = ("Frozen", "NoLangTTT")
NO_LANG = "no lang"
SPLIT_VERSION = "six_action_purge1_disjoint_images_v2"
CACHE_SCHEMA = 3
PURGE_WINDOWS = 1
REGISTERED_SWAP_PAIRS = {0: 1, 1: 0}
REGISTERED_TASK_NAMES = {
    0: "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    1: "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
}
THREAT_MODEL = {
    "attacker_capability_assumption": "Rewrite only pretokenized cached .npy['text']; no access to physical data, actions, targets, policy Goal or deployed parameters directly.",
    "online_sampler_boundary": "Released CALVIN sampler canonicalizes adaptation text to literal 'no lang'; modifying legal policy Goal alone does not alter it.",
    "implementation_scope": "LIBERO reconstruction replays immutable physical data.pt and loads separately serialized token-only .npy metadata with the same dictionary text-field convention. Not the missing official LIBERO trainer.",
    "held_fixed": ["actual action sequence", "observations and future targets", "task and initial state", "LoRA initialization", "training order and RNG", "optimizer and loss", "native deployment mask and original policy Goal"],
    "native_adaptation_text": NO_LANG,
    "goal_ttt_status": "Original task-goal metadata diagnostic, not a native normal-adaptation baseline.",
    "not_proven": ["attacker can write deployment cache", "native online Goal-Swap", "decision damage", "Bus-Cut repair"],
}


def physical_protocol(args):
    return {"suite": args.suite, "task_id": args.task_id, "source_init_id": args.source_init_id,
            "seed": args.seed, "camera_size": args.camera_size,
            "settle_steps": args.settle_steps, "orientation": args.orientation,
            "source_max_steps": args.max_steps, "processed_image_size": 224,
            "env_action_dtype": "float16", "image_rotation_degrees": 0,
            "patch_whitening": False}


def window_starts(num_actions, train_windows=30, heldout_windows=17):
    """Length-only split: fixed prefix, one purged window, then heldout data."""
    if train_windows != 30 or heldout_windows < 1:
        raise ValueError("Require exactly 30 training windows and positive heldout count")
    train = list(range(0, train_windows * 6, 6))
    remaining = list(range((train_windows + PURGE_WINDOWS) * 6, num_actions - 5, 6))
    if len(remaining) < heldout_windows:
        raise ValueError(f"After 30 training windows and one purged window, {num_actions} actions "
                         f"provide only {len(remaining)} heldout windows; requested {heldout_windows}")
    selected = np.linspace(0, len(remaining) - 1, heldout_windows, dtype=int)
    heldout = [remaining[int(i)] for i in selected]
    train_frames = {s + o for s in train for o in (0, 3, 6)}
    heldout_frames = {s + o for s in heldout for o in (0, 3, 6)}
    if train_frames & heldout_frames:
        raise AssertionError("Training/heldout image indices overlap")
    return train, heldout


def split_protocol(num_actions, train_windows=30, heldout_windows=17):
    train, heldout = window_starts(num_actions, train_windows, heldout_windows)
    return {"version": SPLIT_VERSION, "source_action_count": num_actions,
            "selection_basis": "Fixed training prefix and one-window purge; heldout count and positions depend only on trace length, never forecast loss or task/attack effects.",
            "train_starts": train, "purged_starts": [train_windows * 6],
            "heldout_starts": heldout, "window_action_count": 6,
            "input_offsets": [0, 3], "target_offsets": [3, 6],
            "train_frame_indices": sorted({s + o for s in train for o in (0, 3, 6)}),
            "heldout_frame_indices": sorted({s + o for s in heldout for o in (0, 3, 6)}),
            "cross_partition_images_disjoint": True,
            "statistical_unit": "One source trajectory; adjacent within-partition windows are correlated and may share boundary images. Heldout windows are not independent episodes."}


def replay_data(model, task, source_row, records, provenance, args, gate):
    """Reuse the original exact-control collector, then discard purge data.

    The collector's training prefix is increased by one only to collect the
    required later heldout observations. It performs no optimization. The
    extra window's interior frame is discarded before cache validation; the
    saved training set remains exactly 30 windows. No core globals are changed.
    """
    train, heldout = window_starts(len(records), args.train_windows, args.heldout_windows)
    collector_args = SimpleNamespace(**vars(args))
    collector_args.train_windows += PURGE_WINDOWS
    cache = core.replay_data(model, task, source_row, records, provenance, collector_args, gate)
    if (cache["train_starts"] != train + [args.train_windows * 6]
            or cache["heldout_starts"] != heldout):
        raise RuntimeError("Legacy collector no longer matches the registered purged split")
    frames = sorted({s + o for s in train + heldout for o in (0, 3, 6)})
    lookup = {frame: i for i, frame in enumerate(cache["frame_indices"])}
    keep = torch.tensor([lookup[frame] for frame in frames], dtype=torch.int64)
    cache["images"] = cache["images"].index_select(0, keep)
    cache["states"] = cache["states"].index_select(0, keep)
    cache.update(schema=CACHE_SCHEMA, physical_protocol=physical_protocol(args),
                 split_protocol=split_protocol(len(records), args.train_windows, args.heldout_windows),
                 train_starts=train, heldout_starts=heldout, frame_indices=frames,
                 data_split="30 six-action training windows; one purged six-action window; later same-trajectory heldout windows. No cross-partition image/target indices overlap; not independent-trajectory validation.")
    return cache


def validate_source(args):
    row, records, provenance = core.validate_source(args)
    source = json.loads(args.source_summary.read_text())
    if source["protocol"].get("max_steps") != args.max_steps:
        raise ValueError("Source horizon differs from requested protocol")
    return row, records, provenance


def physical_bddl_sections(path):
    text = path.read_text()
    result = {}
    for name in ("problem", ":domain", ":regions", ":fixtures", ":objects", ":init"):
        matches = list(re.finditer(r"\(" + re.escape(name) + r"\s", text))
        if len(matches) != 1:
            raise ValueError(f"Expected one BDDL {name} section in {path}")
        start, depth = matches[0].start(), 0
        for end in range(start, len(text)):
            depth += (text[end] == "(") - (text[end] == ")")
            if depth == 0:
                result[name] = " ".join(text[start:end + 1].split())
                break
        else:
            raise ValueError(f"Unbalanced BDDL {name} section")
    return result


def validate_same_scene_pair(task_id, task, swapped_task, libero_root):
    if task_id not in REGISTERED_SWAP_PAIRS:
        raise ValueError("This pilot only registers same-scene task pairs 0 -> 1 and 1 -> 0")
    swap_id = REGISTERED_SWAP_PAIRS[task_id]
    if task.name != REGISTERED_TASK_NAMES[task_id] or swapped_task.name != REGISTERED_TASK_NAMES[swap_id]:
        raise ValueError("Benchmark task ordering differs from the registered physical pair")
    paths = [libero_root / "libero/libero/bddl_files" / t.problem_folder / t.bddl_file
             for t in (task, swapped_task)]
    left, right = [physical_bddl_sections(path) for path in paths]
    if left != right:
        raise ValueError("Registered pair no longer has matching BDDL physical metadata")
    return {"registered_mapping": REGISTERED_SWAP_PAIRS, "source_task_id": task_id,
            "swap_task_id": swap_id, "scene": "LIVING_ROOM_SCENE2",
            "selection_basis": "Prespecified physical BDDL metadata and same scene, not attack or task outcomes; 1 -> 0 has not been executed by this registration.",
            "matched_sections": list(left), "physical_metadata_sha256": json_sha(left),
            "bddl_files": [{"path": str(path), "sha256": file_sha(path)} for path in paths],
            "scope": "Physical task specification matches; not a claim that separate task init files contain identical numerical states. All arms use the one fixed source state/trajectory.",
            "allowed_task_metadata_differences": [":language", ":obj_of_interest", ":goal"]}


def validate_cache(cache, args, records, provenance, clean_goal):
    """Reject mismatched render protocols even though CLIP always outputs 224px."""
    if cache.get("schema") != CACHE_SCHEMA:
        raise ValueError("Require physical cache schema 3 with purged image-disjoint split; legacy schema 1/2 caches are diagnostic artifacts and cannot be reused or silently relabeled")
    if cache.get("source") != provenance or cache.get("clean_goal") != clean_goal:
        raise ValueError("Cache source/Goal differs from requested source")
    expected_protocol = physical_protocol(args)
    if cache.get("physical_protocol") != expected_protocol:
        raise ValueError("Cached physical protocol differs (including camera resolution)")
    expected_split = split_protocol(len(records), args.train_windows, args.heldout_windows)
    if cache.get("split_protocol") != expected_split:
        raise ValueError("Cached split protocol differs from registered purge/version")
    train, heldout = window_starts(len(records), args.train_windows, args.heldout_windows)
    frames = sorted({start + offset for start in train + heldout for offset in (0, 3, 6)})
    if cache.get("train_starts") != train or cache.get("heldout_starts") != heldout:
        raise ValueError("Cache train/heldout window split differs")
    if cache.get("frame_indices") != frames:
        raise ValueError("Cache frame indices differ from the physical window specification")
    expected_actions = torch.tensor([row["action"] for row in records], dtype=torch.float32)
    if cache["actions"].dtype != torch.float32 or not torch.equal(cache["actions"], expected_actions):
        raise ValueError("Cached action values differ from source trace")
    if tuple(cache["images"].shape) != (len(frames), 2, 3, 224, 224):
        raise ValueError("Cached processed image dimensions differ")
    if tuple(cache["states"].shape) != (len(frames), 7):
        raise ValueError("Cached state dimensions differ")
    if not all(bool(torch.isfinite(cache[key]).all()) for key in ("images", "states", "actions")):
        raise ValueError("Nonfinite cached physical data")
    replay = cache.get("replay", {})
    if (not replay.get("reward_done_success_exact") or replay.get("checked_steps") != max(frames)
            or max(replay.get("max_abs", {"missing": float("inf")}).values()) > args.replay_atol):
        raise ValueError("Cache lacks matching fixed-control replay validation")
    return {"schema": cache["schema"], "physical_protocol": expected_protocol,
            "split_protocol": expected_split, "legacy_cache_accepted": False,
            "source_actions_exact": True, "frame_window_alignment_exact": True}


def check_clean_gate(results):
    if not all(name in results for name in HEALTH_CONDITIONS):
        return {"status": "not_yet_evaluated", "passed": False}
    mapped = {"Frozen": results["Frozen"], "CleanTTT": results["NoLangTTT"]}
    gate = core.check_clean_gate(mapped)
    gate["baseline_condition"] = "NoLangTTT"
    gate["baseline_text"] = NO_LANG
    gate["no_lang_ttt_successes"] = gate.pop("clean_ttt_successes", None)
    gate["criterion"] = ("At least one Frozen success and no Frozen-success episode lost after "
                         "NoLangTTT; development health gate, not statistical equivalence.")
    return gate


def validate_native_gate(path, args):
    if path is None:
        raise ValueError("Diagnostic/poisoning arms require --native-task-gate-summary")
    source = json.loads(path.read_text())
    protocol = source.get("protocol", {})
    if source.get("status") != "completed":
        raise ValueError("Native task gate is incomplete")
    for key, expected in {"suite": "libero_10", "camera_size": args.camera_size,
                          "seed": args.seed, "orientation": args.orientation,
                          "settle_steps": args.settle_steps, "max_steps": args.max_steps}.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Native task gate {key} does not match")
    rows = [row for row in source.get("results", []) if row.get("condition") == "native"]
    expected = {(task, init) for task in range(10) for init in range(3)}
    if len(rows) != 30 or {(r["task_id"], r["init_id"]) for r in rows} != expected:
        raise ValueError("Native clean gate must include all fixed 10 tasks x 3 initial states")
    if any(not isinstance(row.get("success"), bool) for row in rows):
        raise ValueError("Native gate success values must be JSON booleans; strings/numbers are invalid")
    if any(not isinstance(row.get("success_at_start"), bool) for row in rows):
        raise ValueError("Native gate success_at_start values must be JSON booleans")
    if any(row.get("success_at_start", True) for row in rows):
        raise ValueError("Native clean gate contains trivial or unverified starting success")
    successes = sum(row["success"] for row in rows)
    successful_tasks = len({row["task_id"] for row in rows if row["success"]})
    if successes < 21 or successful_tasks < 7:
        raise ValueError(f"Native clean gate failed: {successes}/30 successes, {successful_tasks}/10 tasks")
    return {"path": str(path.resolve()), "sha256": file_sha(path), "successes": successes,
            "episodes": 30, "successful_tasks": successful_tasks, "passed": True,
            "criterion": "At least 21/30 successes and at least 7/10 tasks; fixed all-task protocol"}


def serialize_metadata(tokenizer, clean_goal, swap_goal, device, output):
    """Separate token-only files model the native loader's cached text field.

    They do not claim to be complete original CALVIN trajectory cache files.
    Their physical data are the independently validated, immutable data.pt.
    """
    values = {"no_lang": NO_LANG, "goal": clean_goal, "swap": swap_goal, "empty": ""}
    tokens, metadata = {}, {}
    for name, text in values.items():
        encoded = tokenizer([text], truncate=True).unsqueeze(1).repeat(1, 2, 1).cpu()
        path = output / f"metadata_{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, {"text": encoded.numpy()})
        # This matches the public loader's dictionary/text convention. Only
        # files created in this run are read; no untrusted pickle is required.
        actual = torch.from_numpy(np.load(path, allow_pickle=True).item()["text"])
        if (actual.dtype not in (torch.int32, torch.int64) or actual.dtype != encoded.dtype
                or actual.shape != (1, 2, 77) or not torch.equal(actual, encoded)):
            raise RuntimeError("Serialized metadata changed token values or shape")
        tokens[name] = actual.to(device)
        metadata[name] = {"literal_text": text, "file": str(path), "sha256": file_sha(path),
                          "token_sha256": tensor_digest({"text": actual}), "shape": list(actual.shape)}
    return tokens, metadata


def self_test():
    import tempfile
    from unittest.mock import patch
    core.self_test()
    args = SimpleNamespace(suite="libero_10", task_id=0, source_init_id=0, seed=31,
                           camera_size=128, settle_steps=5, orientation="euler", max_steps=600,
                           train_windows=30, heldout_windows=17, replay_atol=1e-9)
    records = [{"action": [0.] * 6 + [-1.]} for _ in range(293)]
    train, heldout = window_starts(len(records), 30, 17)
    assert train == list(range(0, 180, 6)) and heldout == list(range(186, 288, 6))
    assert not {s + o for s in train for o in (0, 3, 6)} & {s + o for s in heldout for o in (0, 3, 6)}
    for unsupported in (18, 20):
        try:
            window_starts(293, 30, unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError("Oversized heldout split accepted for 293-action trajectory")
    frames = sorted({s + o for s in train + heldout for o in (0, 3, 6)})
    cache = {"schema": CACHE_SCHEMA, "source": {"summary_sha256": "matched"}, "clean_goal": "original",
             "physical_protocol": physical_protocol(args), "train_starts": train,
             "split_protocol": split_protocol(len(records), 30, 17),
             "heldout_starts": heldout, "frame_indices": frames,
             "actions": torch.tensor([r["action"] for r in records]),
             "images": torch.zeros(1).expand(len(frames), 2, 3, 224, 224),
             "states": torch.zeros(len(frames), 7),
             "replay": {"reward_done_success_exact": True, "checked_steps": max(frames), "max_abs": {"eef": 0.}}}
    validate_cache(cache, args, records, cache["source"], "original")
    for legacy_schema in (1, 2):
        try:
            validate_cache(dict(cache, schema=legacy_schema), args, records, cache["source"], "original")
        except ValueError as error:
            assert "legacy" in str(error)
        else:
            raise AssertionError("Legacy split cache accepted")
    try:
        validate_cache(dict(cache, split_protocol={}), args, records, cache["source"], "original")
    except ValueError:
        pass
    else:
        raise AssertionError("Unversioned split accepted")
    def fake_collector(model, task, row, actual_records, provenance, collector_args, gate):
        assert collector_args.train_windows == 31 and actual_records is records
        collected_train, collected_heldout = core.window_starts(len(records), 31, 17)
        collected_frames = sorted({s + o for s in collected_train + collected_heldout for o in (0, 3, 6)})
        values = torch.tensor(collected_frames)[:, None]
        return {"train_starts": collected_train, "heldout_starts": collected_heldout,
                "frame_indices": collected_frames, "images": values, "states": values.clone()}
    with patch.object(core, "replay_data", fake_collector):
        collected = replay_data(None, None, None, records, {}, args, None)
    assert args.train_windows == 30
    assert collected["images"][:, 0].tolist() == frames
    assert collected["states"][:, 0].tolist() == frames
    assert 183 not in collected["frame_indices"] and collected["train_starts"] == train
    assert collected["heldout_starts"] == heldout and collected["schema"] == CACHE_SCHEMA
    args.camera_size = 224
    try:
        validate_cache(cache, args, records, cache["source"], "original")
    except ValueError:
        pass
    else:
        raise AssertionError("128-camera cache accepted into 224-camera protocol")
    args.camera_size = 128
    healthy = {name: {"episodes": [{"init_id": 0, "success": True}]} for name in HEALTH_CONDITIONS}
    assert check_clean_gate(healthy)["passed"]
    assert not check_clean_gate({"Frozen": healthy["Frozen"], "GoalTTT": healthy["NoLangTTT"]})["passed"]
    healthy["NoLangTTT"]["episodes"][0]["success"] = False
    assert not check_clean_gate(healthy)["passed"]
    from clip import tokenize
    assert not torch.equal(tokenize([NO_LANG]), tokenize([""]))
    with tempfile.TemporaryDirectory() as temporary:
        tokens, metadata = serialize_metadata(tokenize, "original", "alternate", "cpu", Path(temporary))
        assert torch.equal(tokens["no_lang"], tokenize([NO_LANG]).unsqueeze(1).repeat(1, 2, 1))
        assert metadata["no_lang"]["literal_text"] == "no lang"
        p = Path(temporary) / "native_gate.json"
        rows = [{"task_id": t, "init_id": i, "condition": "native", "success_at_start": False,
                 "success": t < 7} for t in range(10) for i in range(3)]
        protocol = {"suite": "libero_10", "camera_size": 128, "seed": 31,
                    "orientation": "euler", "settle_steps": 5, "max_steps": 600}
        write_json(p, {"status": "completed", "protocol": protocol, "results": rows})
        assert validate_native_gate(p, args)["successes"] == 21
        args.camera_size = 224
        try:
            validate_native_gate(p, args)
        except ValueError as error:
            assert "camera_size" in str(error)
        else:
            raise AssertionError("128 clean gate accepted into 224 protocol")
        args.camera_size = 128
        rows[0]["success"] = "false"
        write_json(p, {"status": "completed", "protocol": protocol, "results": rows})
        try:
            validate_native_gate(p, args)
        except ValueError as error:
            assert "booleans" in str(error)
        else:
            raise AssertionError("String success accepted as boolean")
        rows[0]["success"] = False
        write_json(p, {"status": "completed", "protocol": protocol, "results": rows})
        try:
            validate_native_gate(p, args)
        except ValueError:
            pass
        else:
            raise AssertionError("Below-threshold clean gate accepted")
    tasks = {i: SimpleNamespace(name=name, problem_folder="libero_10", bddl_file=name + ".bddl")
             for i, name in REGISTERED_TASK_NAMES.items()}
    for task_id, swap_id in REGISTERED_SWAP_PAIRS.items():
        pair = validate_same_scene_pair(task_id, tasks[task_id], tasks[swap_id], ROOT / "vendor/LIBERO")
        assert pair["swap_task_id"] == swap_id
    try:
        validate_same_scene_pair(2, tasks[0], tasks[1], ROOT / "vendor/LIBERO")
    except ValueError:
        pass
    else:
        raise AssertionError("Unregistered task pair accepted")
    print(json.dumps({"status": "passed", "device": "cpu", "gpu_inference": False,
                      "checks": ["literal no lang differs from empty", "cached metadata token roundtrip",
                                 "128/224 cache mismatch rejected", "GoalTTT cannot substitute for NoLangTTT gate",
                                 "lost Frozen success rejects adaptation gate", "all-task 21/30 gate enforced",
                                 "string success rejected", "registered 0<->1 physical BDDL sections match",
                                 "unregistered task pairs rejected", "293 actions give 30 train/purge1/17 heldout",
                                 "no cross-partition image indices", "18/20 heldout rejected for length",
                                 "legacy/unversioned split cache rejected", "collector purge filtering without render",
                                 "128/224 native clean-gate mismatch rejected"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true",
                        help="Replay fixed source controls and cache CPU-processed RGB; no model load/forward, LoRA or episodes")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-summary", type=Path, default=SOURCE)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--normal-gate-summary", "--clean-gate-summary", dest="clean_gate_summary", type=Path)
    parser.add_argument("--native-task-gate-summary", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(HEALTH_CONDITIONS))
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
    parser.add_argument("--heldout-windows", type=int, default=17)
    parser.add_argument("--replay-atol", type=float, default=1e-9)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for experiments")
    if args.train_windows != 30 or args.heldout_windows < 1:
        parser.error("This registered single-update pilot requires exactly 30 training windows")
    if args.camera_size not in (128, 224) or args.max_steps < 1 or args.replay_atol <= 0 or args.threads < 1:
        parser.error("Require 128- or 224-pixel rendering and positive evaluation/replay limits")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Duplicate conditions")
    args.eval_init_ids = [int(item) for item in args.eval_init_ids.split(",")]
    if len(set(args.eval_init_ids)) != len(args.eval_init_ids) or not args.eval_init_ids or min(args.eval_init_ids) < 0:
        parser.error("Invalid evaluation initial-state IDs")
    if (not args.prepare_data_only and any(c not in HEALTH_CONDITIONS for c in args.conditions)
            and not args.clean_gate_summary):
        if args.conditions[:2] != list(HEALTH_CONDITIONS):
            parser.error("Run Frozen then NoLangTTT first, or supply a matching passed --normal-gate-summary")
    if args.suite != "libero_10":
        parser.error("This registered pilot is limited to libero_10")
    if args.task_id not in REGISTERED_SWAP_PAIRS:
        parser.error("This pilot only supports prespecified same-scene task pairs 0 -> 1 and 1 -> 0")
    if args.swap_task_id is not None and args.swap_task_id != REGISTERED_SWAP_PAIRS[args.task_id]:
        parser.error("Swap must follow the metadata-selected registered mapping {0: 1, 1: 0}")
    extended = not args.prepare_data_only and any(c not in HEALTH_CONDITIONS for c in args.conditions)
    native_gate = validate_native_gate(args.native_task_gate_summary, args) if extended else None
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
    data_split = split_protocol(len(records), args.train_windows, args.heldout_windows)
    suite = gate.base.benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    swap_id = REGISTERED_SWAP_PAIRS[args.task_id]
    swapped_task = suite.get_task(swap_id)
    same_scene_pair = validate_same_scene_pair(args.task_id, task, swapped_task, gate.base.LIBERO_ROOT)
    swap_goal = swapped_task.language
    protocol = {"description": __doc__, "script_sha256": file_sha(__file__), "threat_model": THREAT_MODEL, "source": source_provenance,
                "suite": args.suite, "task_id": args.task_id, "source_init_id": args.source_init_id,
                "eval_init_ids": args.eval_init_ids, "swap_task_id": swap_id,
                "clean_goal": task.language, "swap_goal": swap_goal,
                "normal_adaptation_text": NO_LANG,
                "normal_adaptation_conditions": list(HEALTH_CONDITIONS),
                "same_scene_swap_registration": same_scene_pair,
                "environment_seed": args.seed, "adapter_seed": args.adapter_seed,
                "model_device": args.device, "torch_threads": args.threads,
                "cpu_scope": "Separate CPU Frozen/TTT baseline; CUDA source actions used only for fixed-data replay. "
                             "CPU and CUDA success/gradient results must not be pooled as paired conditions.",
                "max_steps": args.max_steps, "camera_size": args.camera_size,
                "orientation": args.orientation, "settle_steps": args.settle_steps,
                "train_windows": args.train_windows, "heldout_windows": args.heldout_windows,
                "data_split": data_split,
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
                "not_claimed": ["exact official LIBERO TTT reproduction", "native external attacker access", "native online Goal-Swap", "forecast consumed by frozen policy"],
                "checkpoint_sha256": json.loads((gate.base.MODEL_DIR / "provenance.json").read_text())["sha256"],
                "source_files": {str(path.relative_to(ROOT)): file_sha(path) for path in [
                    Path(__file__), Path(core.__file__), Path(gate.__file__), Path(gate.base.__file__),
                    ROOT / "vendor/WorldAgen/test_time_training/ttt.py",
                    ROOT / "vendor/WorldAgen/test_time_training/lora.py",
                    ROOT / "vendor/WorldAgen/scripts/libero/eval_ttt.sh"]}}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    results = {}
    summary = {"status": "preparing", "protocol": protocol,
               "conditions": [] if args.prepare_data_only else args.conditions,
               "prepare_data_only": args.prepare_data_only,
               "api_requests": 0, "native_task_gate": native_gate, "results": results, "normal_adaptation_gate": {"passed": False, "status": "not_yet_evaluated"}}
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
            cache_validation = validate_cache(cache, args, records, source_provenance, task.language)
        else:
            cache = replay_data(processor, task, source_row, records, source_provenance, args, gate)
            cache_validation = validate_cache(cache, args, records, source_provenance, task.language)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("xb") as stream:
                torch.save(cache, stream)
        summary["data"] = {"path": str(cache_path), "sha256": file_sha(cache_path),
                           "validation": cache_validation, "replay": cache["replay"], "train_starts": cache["train_starts"],
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
        # A staged arm must use the same implementation as its NoLangTTT
        # health gate; a later code change requires revalidating that gate.
        identity = {key: value for key, value in protocol.items() if key != "description"}
        identity["cache_sha256"] = summary["data"]["sha256"]
        summary["protocol_identity_sha256"] = json_sha(identity)
        if args.clean_gate_summary:
            prior = json.loads(args.clean_gate_summary.read_text())
            if (prior.get("status") != "completed"
                    or not prior.get("normal_adaptation_gate", {}).get("passed")
                    or prior.get("protocol_identity_sha256") != summary["protocol_identity_sha256"]):
                raise RuntimeError("Prior NoLangTTT health gate did not pass or protocol/data identity differs")
            summary["normal_adaptation_gate"] = prior["normal_adaptation_gate"]
            summary["normal_gate_source"] = {"path": str(args.clean_gate_summary.resolve()), "sha256": file_sha(args.clean_gate_summary)}
        model = gate.base.build_model()
        native, bus, control = make_masks(model, gate)
        texts, metadata = serialize_metadata(gate.base.clip.tokenize, task.language, swap_goal,
                                             gate.base.DEVICE, output)
        summary["cached_metadata"] = metadata
        before_attach = capture_zero_adapter_reference(model, cache, texts["goal"], native)
        seed_all(args.adapter_seed, cuda=args.device == "cuda")
        adapted_names = attach_lora(model)
        after_attach = capture_zero_adapter_reference(model, cache, texts["goal"], native)
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
        arms = {"Frozen": ("no_lang", native), "NoLangTTT": ("no_lang", native),
                "GoalTTT": ("goal", native), "SwapTTT": ("swap", native),
                "EmptyTTT": ("empty", native), "NoLangBusCutTTT": ("no_lang", bus),
                "SwapBusCutTTT": ("swap", bus), "SwapControlCutTTT": ("swap", control)}
        for condition in args.conditions:
            if condition not in HEALTH_CONDITIONS and not summary["normal_adaptation_gate"]["passed"]:
                summary["status"] = "stopped_no_lang_ttt_gate_failed"
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
                      "adaptation_literal": metadata[text_name]["literal_text"],
                      "adaptation_metadata_sha256": metadata[text_name]["sha256"],
                      "training": training, "heldout_before_adaptation_condition": pre,
                      "heldout_after_adaptation_condition": evaluate_forecasts(model, cache, cache["heldout_starts"], texts[text_name], mask),
                      "heldout_after_original_goal_native": evaluate_forecasts(model, cache, cache["heldout_starts"], texts["goal"], native),
                      "heldout_after_no_lang_native": evaluate_forecasts(model, cache, cache["heldout_starts"], texts["no_lang"], native),
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
            if all(name in results for name in HEALTH_CONDITIONS):
                summary["normal_adaptation_gate"] = check_clean_gate(results)
            summary["elapsed_seconds"] = time.monotonic() - started
            write_json(summary_path, summary)
        else:
            summary["status"] = "completed"
        summary["causal_comparisons"] = {
            "no_lang_vs_swap_cut": compare_artifacts(results, "NoLangBusCutTTT", "SwapBusCutTTT", output),
            "swap_vs_matched_control": compare_artifacts(results, "SwapTTT", "SwapControlCutTTT", output)}
        summary["aggregate"] = {name: {"success_rate": row["task_success_rate"], "successes": row["task_successes"],
                                      "episodes": len(row["episodes"])} for name, row in results.items()}
        summary["scientific_scope"] = (
            "Development evidence only; no attack/repair inference unless actual success losses/recovery occur. "
            "NoLangTTT is the sampler-matched canonicalization baseline; EmptyTTT uses a different literal. "
            "GoalTTT and SwapTTT are cached-metadata interventions under assumed cache-writer access, "
            "not attacks through a native legal-Goal adaptation interface.")
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
                      "normal_adaptation_gate": summary["normal_adaptation_gate"], "aggregate": summary.get("aggregate")}), flush=True)


if __name__ == "__main__":
    main()
