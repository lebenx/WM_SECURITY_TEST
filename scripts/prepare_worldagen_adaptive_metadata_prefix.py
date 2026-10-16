#!/usr/bin/env python3
"""Export only the already registered 30-window training prefix, without inference.

This is procedural data isolation, not an OS access-control boundary. The output
contains no post-prefix actions, heldout images/indices, rewards or task outcomes.
The provenance sidecar is for the auditor, not input to the selection program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs/worldagen_cached_metadata_camera128_data_v1/data.pt"
CACHE_SHA = "1bdb68646c5f9d7bb13d26744745a96723b5b10509ea8dbf305bc22ab8f8c05f"
INITIAL = ROOT / "runs/worldagen_cached_metadata_camera128_health_v1/initial_adapter.pt"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Output must be empty; existing artifacts are never replaced")
    if sha(CACHE) != CACHE_SHA:
        raise RuntimeError("Immutable physical cache hash differs")
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    starts = list(range(0, 180, 6))
    frames = list(range(0, 181, 3))
    assert cache["schema"] == 3 and cache["train_starts"] == starts
    assert cache["physical_protocol"]["camera_size"] == 128
    assert not set(frames) & set(cache["split_protocol"]["heldout_frame_indices"])
    lookup = {v: i for i, v in enumerate(cache["frame_indices"])}
    indices = torch.tensor([lookup[f] for f in frames])
    prefix = {
        "train_starts": starts,
        "frame_indices": frames,
        "images": cache["images"].index_select(0, indices).contiguous(),
        "states": cache["states"].index_select(0, indices).contiguous(),
        "actions": cache["actions"][:180].clone().contiguous(),
    }
    assert prefix["images"].shape == (61, 2, 3, 224, 224)
    assert prefix["states"].shape == (61, 7)
    assert prefix["actions"].shape == (180, 7)
    # Clones prevent torch.save from serializing a larger backing storage.
    for key in ("images", "states", "actions"):
        x = prefix[key]
        assert x.untyped_storage().nbytes() == x.numel() * x.element_size()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prefix.pt").open("xb") as f:
        torch.save(prefix, f)
    shutil.copyfile(INITIAL, out / "initial_adapter.pt")
    check = torch.load(out / "prefix.pt", map_location="cpu", weights_only=True)
    for key in prefix:
        assert torch.equal(check[key], prefix[key]) if torch.is_tensor(prefix[key]) else check[key] == prefix[key]
    manifest = {
        "status": "prefix_export_verified_no_model_or_environment_calls",
        "preparation_script_sha256": sha(__file__),
        "source_cache": {"path": str(CACHE), "sha256": CACHE_SHA},
        "source_initial_adapter": {"path": str(INITIAL), "sha256": sha(INITIAL)},
        "prefix_cache": {"path": str(out / "prefix.pt"), "sha256": sha(out / "prefix.pt")},
        "initial_adapter": {"path": str(out / "initial_adapter.pt"), "sha256": sha(out / "initial_adapter.pt")},
        "allowed_keys": list(prefix), "train_windows": 30, "frames": 61,
        "action_indices": [0, 179], "last_frame": 180,
        "contains_heldout_or_task_outcomes": False,
        "data_isolation": "Selection program receives prefix.pt and copied initial adapter only; not an OS sandbox. This auditor sidecar is not selection input.",
        "api_requests": 0,
    }
    (out / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
