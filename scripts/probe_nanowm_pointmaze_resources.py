#!/usr/bin/env python3
"""CPU-only NanoWM PointMaze resource audit; no model or planner execution."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/nano-world-model"
REVISION = "cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0"
MODEL_REVISION = "b84303b7151cbd1836841c3a87bf095e1015b150"
ARCHIVE_URL = "https://osf.io/download/vr5gy/"
ARCHIVE_SIZE = 718363945
CONFIG_URL = ("https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k/"
              f"resolve/{MODEL_REVISION}/config.yaml")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RangeFile(io.RawIOBase):
    """Seekable bounded HTTP reader. Never accepts a whole-archive response."""
    def __init__(self):
        self.pos = 0
        self.bytes_received = 0
        self.requests = []
        self.download_url = ARCHIVE_URL

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else ARCHIVE_SIZE + offset
        if not 0 <= self.pos <= ARCHIVE_SIZE:
            raise ValueError("Invalid archive seek")
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = ARCHIVE_SIZE - self.pos
        size = min(size, ARCHIVE_SIZE - self.pos)
        if not size:
            return b""
        if size > 16 * 1024**2 or self.bytes_received + size > 64 * 1024**2:
            raise RuntimeError("Metadata-only transfer budget exceeded")
        start, end = self.pos, self.pos + size - 1
        for attempt in range(3):
            req = urllib.request.Request(self.download_url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=45) as response:
                    expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Range response mismatch; full archive download refused")
                    data = response.read(size + 1)
                    # Reuse the official redirect during this transfer; do not persist signed URLs.
                    self.download_url = response.geturl()
                break
            except urllib.error.HTTPError as error:
                if error.code not in (403, 429, 500, 502, 503, 504) or attempt == 2:
                    raise
                self.download_url = ARCHIVE_URL + f"?metadata_probe={time.time_ns()}"
                time.sleep(0.5 * (attempt + 1))
        if len(data) != size:
            raise RuntimeError("Range response size mismatch")
        self.pos += size
        self.bytes_received += size
        self.requests.append({"start": start, "end": end, "bytes": size})
        return data


def fetch_metadata(out):
    data_dir = out / "data/point_maze"
    data_dir.mkdir(parents=True)
    remote = RangeFile()
    records = []
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        for basename in ("actions.pth", "seq_lengths.pth"):
            matches = [name for name in names if Path(name).name == basename and "__MACOSX" not in name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one archive member for {basename}: {matches}")
            info = archive.getinfo(matches[0])
            if info.file_size > 32 * 1024**2 or info.compress_size > 16 * 1024**2:
                raise RuntimeError("Metadata member is too large")
            payload = archive.read(info)
            path = data_dir / basename
            path.write_bytes(payload)
            records.append({"archive_member": info.filename, "path": str(path.relative_to(ROOT)),
                            "bytes": len(payload), "sha256": sha(path), "zip_crc32": info.CRC})
    config_path = out / "checkpoint_config.yaml"
    with urllib.request.urlopen(CONFIG_URL, timeout=30) as response:
        payload = response.read(65537)
    if len(payload) > 65536:
        raise RuntimeError("Unexpectedly large model configuration")
    config_path.write_bytes(payload)
    return data_dir, {"archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_SIZE,
                      "downloaded_bytes": remote.bytes_received, "range_requests": remote.requests,
                      "members": records, "video_members_downloaded": 0,
                      "config_url": CONFIG_URL, "config_sha256": sha(config_path)}


def native_stats(data_dir, out):
    """Execute unmodified native split/stat methods without visual data imports."""
    import numpy as np
    import torch
    import yaml
    cfg = yaml.safe_load((out / "checkpoint_config.yaml").read_text())
    loader = cfg["dataset"]["loader"]
    assert loader["random_seed"] == 42 and loader["split_ratio"] == 0.9
    assert loader["n_rollout"] is None and loader["normalize_action"]
    assert cfg["dataset"]["frame_interval"] == 5
    actions = torch.load(data_dir / "actions.pth", map_location="cpu", weights_only=True).float()
    lengths = torch.as_tensor(torch.load(data_dir / "seq_lengths.pth", map_location="cpu", weights_only=True))
    assert actions.ndim == 3 and actions.shape[-1] == 2
    assert len(lengths) == len(actions) and bool((lengths > 0).all())
    assert bool((lengths <= actions.shape[1]).all()) and bool(torch.isfinite(actions).all())
    scale = float(loader.get("action_scale") or 1.0)
    actions = actions / scale
    source = types.SimpleNamespace(data_path=data_dir, num_trajectories=len(actions), action_dim=2)
    source.load_trajectory = lambda index: types.SimpleNamespace(actions=actions[index], seq_length=int(lengths[index]))

    cache_path = VENDOR / "src/wm_datasets/stats_cache.py"
    tree = ast.parse(cache_path.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.level and node.module == "data_source")]
    cache_module = types.ModuleType("nanowm_native_stats_cache")
    cache_module.__dict__["DataSource"] = object
    exec(compile(tree, str(cache_path), "exec"), cache_module.__dict__)

    dataset_path = VENDOR / "src/wm_datasets/world_model_dataset.py"
    tree = ast.parse(dataset_path.read_text())
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorldModelDataset")
    wanted = {"_split_trajectories_indices", "_compute_action_stats", "_broadcast_action_stats", "_split_indices", "_cache_path"}
    methods = [node for node in native_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in methods} == wanted
    minimal = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              ast.ClassDef(name="NativeStatsOnly", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(minimal)
    namespace = {"torch": torch, "np": np, "stats_cache": cache_module}
    exec(compile(minimal, str(dataset_path), "exec"), namespace)
    obj = namespace["NativeStatsOnly"]()
    obj.data_source, obj.num_trajectories = source, len(actions)
    obj.random_seed, obj.split_ratio, obj.split = loader["random_seed"], loader["split_ratio"], "train"
    obj.frame_interval, obj.use_data_source_stats = 5, True
    obj._split_trajectories_indices()
    obj._compute_action_stats()
    train_ids = list(obj.traj_indices)
    obj.split = "val"
    obj._split_trajectories_indices()
    val_ids = list(obj.traj_indices)
    assert not set(train_ids).intersection(val_ids)
    assert sorted(train_ids + val_ids) == list(range(len(actions)))
    independent = torch.cat([actions[i, :int(lengths[i])] for i in train_ids], dim=0)
    assert torch.equal(obj._raw_action_mean, independent.mean(0))
    assert torch.equal(obj._raw_action_std, independent.std(0) + 1e-6)
    assert bool(torch.isfinite(obj.action_std).all()) and bool((obj.action_std > 0).all())
    caches = list((data_dir / "wm_stats_cache").glob("action_train_*.pt"))
    assert len(caches) == 1
    split_record = {"seed": obj.random_seed, "ratio": obj.split_ratio, "train_ids": train_ids,
                    "validation_ids": val_ids, "disjoint": True,
                    "source": "Unmodified native trajectory split function extracted with AST"}
    write_json(out / "trajectory_split.json", split_record)
    return {"status": "exact_native_training_statistics_recomputed", "num_trajectories": len(actions),
            "train_trajectories": len(train_ids), "validation_trajectories": len(val_ids),
            "train_valid_action_count": len(independent), "raw_mean": obj._raw_action_mean.tolist(),
            "raw_std": obj._raw_action_std.tolist(), "packed_mean": obj.action_mean.tolist(),
            "packed_std": obj.action_std.tolist(), "frame_interval": 5, "action_scale": scale,
            "cache_path": str(caches[0].relative_to(ROOT)), "cache_sha256": sha(caches[0]),
            "cache_count": len(caches), "fail_closed_if_cache_missing_or_ambiguous": True,
            "statistics_source_sha256": sha(dataset_path), "cache_source_sha256": sha(cache_path),
            "split_path": str((out / "trajectory_split.json").relative_to(ROOT)),
            "native_dset_goal_sampling": "Does not apply this trajectory split for single-directory PointMaze; not a held-out evaluation",
            "training_run_binding_limit": "Reconstructed from published config and pinned native code; original author training cache was not published or compared"}


def environment_probe(out):
    import numpy as np
    import torch
    from PIL import Image
    from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler
    isolate_d4rl_offline_env()
    prepare_legacy_compiler()
    sys.path.insert(0, str(VENDOR / "src"))
    from environments.pointmaze.point_maze_wrapper import PointMazeWrapper
    from environments.pointmaze.maze_model import U_MAZE
    env = PointMazeWrapper(maze_spec=U_MAZE, reward_type="sparse", reset_target=False)
    # A resource-only state and action sequence, not any registered evaluation seed/goal.
    start = np.array([0.8, 2.8, 0.0, 0.0])
    actions = np.tile(np.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]]), (3, 1))
    traces, frames = [], []
    try:
        for replay in range(2):
            obs, state = env.prepare(0, start.copy())
            assert obs["visual"].shape == (224, 224, 3) and obs["visual"].dtype == np.uint8
            current_frames = [obs["visual"].copy()]
            rows = [{"state": np.asarray(state).tolist(), "time": float(env.sim.data.time)}]
            for act in actions:
                obs, reward, done, info = env.step(act.copy())
                assert np.isfinite(info["state"]).all()
                current_frames.append(obs["visual"].copy())
                rows.append({"state": np.asarray(info["state"]).tolist(), "reward": float(reward),
                             "done": bool(done), "time": float(env.sim.data.time)})
            traces.append(rows)
            frames.append(np.stack(current_frames))
        state_equal = traces[0] == traces[1]
        pixel_equal = np.array_equal(frames[0], frames[1])
        assert state_equal and pixel_equal
        assert not np.array_equal(np.asarray(traces[0][0]["state"]), np.asarray(traces[0][-1]["state"]))
        Image.fromarray(frames[0][0]).save(out / "environment_initial.png")
        Image.fromarray(frames[0][-1]).save(out / "environment_final.png")
        np.savez(out / "environment_replay.npz", actions=actions, frames=frames[0])
        write_json(out / "environment_trace.json", {"seed": 0, "start": start.tolist(), "replays": traces})
        return {"status": "reset_render_step_and_exact_replay_passed", "native_class": str(type(env)),
                "replays": 2, "steps_per_replay": len(actions), "state_replay_equal": state_equal,
                "pixel_replay_equal": pixel_equal, "image_shape": list(frames[0].shape[1:]),
                "renderer": "mujoco-py CPU OSMesa", "cuda_initialized": torch.cuda.is_initialized(),
                "task_success_or_planner_capability_claim": False,
                "goal_sampling": "No evaluation goals sampled; fixed seed 0 resource probe only",
                "reward_scope": "Sparse environment target may be OFF_TARGET; do not use this reward as sampled-goal utility"}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/nanowm_resource_gate_v1")
    parser.add_argument("--download-metadata", action="store_true", help="Fetch only bounded official ZIP metadata members")
    parser.add_argument("--reuse-environment-probe", type=Path,
                        help="Reuse a recorded successful CPU environment probe without rerunning physics")
    args = parser.parse_args()
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cpu")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running_cpu_resource_probe", "model_executed": False, "gpu_used": False,
               "source_revision": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
               "script_sha256": sha(__file__), "environment": None, "statistics": None}
    write_json(out / "summary.json", summary)
    try:
        assert summary["source_revision"] == REVISION
        if args.reuse_environment_probe:
            previous = args.reuse_environment_probe.resolve()
            record = json.loads((previous / "summary.json").read_text())
            assert record["source_revision"] == REVISION
            assert record["environment"]["status"] == "reset_render_step_and_exact_replay_passed"
            assert record["environment"]["state_replay_equal"] and record["environment"]["pixel_replay_equal"]
            assert record["model_executed"] is False and record["gpu_used"] is False
            summary["environment"] = record["environment"]
            summary["environment_reused_from"] = {
                "path": str(previous.relative_to(ROOT)),
                "files": [{"path": str((previous / name).relative_to(ROOT)), "sha256": sha(previous / name)}
                          for name in ("summary.json", "environment_trace.json", "environment_replay.npz",
                                       "environment_initial.png", "environment_final.png")],
            }
        else:
            summary["environment"] = environment_probe(out)
        write_json(out / "summary.json", summary)
        if not args.download_metadata:
            raise RuntimeError("Exact training statistics unavailable: metadata download was not requested")
        data_dir, source = fetch_metadata(out)
        summary["metadata_provenance"] = source
        write_json(out / "summary.json", summary)
        summary["statistics"] = native_stats(data_dir, out)
        summary["status"] = "completed_cpu_resource_probe"
        summary["clean_planner_gate_passed"] = False
    except Exception as error:
        summary["status"] = "failed_closed_cpu_resource_probe"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
