#!/usr/bin/env python3
"""Fetch the two fixed public assets for the registered SafeDreamer gate."""
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/safedreamer_mode_mediator_gate_20261016.json"
cfg = json.loads(CONFIG.read_text())
out = ROOT / cfg["output"]
out.mkdir(parents=True, exist_ok=True)
shutil.copy2(CONFIG, out / "frozen_protocol.json")
started = time.monotonic()
record = {"status": "running", "assets": [], "model_calls": 0,
          "environment_steps": 0, "model_api_calls": 0}

def save():
    record["elapsed_seconds"] = time.monotonic() - started
    (out / "asset_fetch.json").write_text(json.dumps(record, indent=2) + "\n")

def download(url, path, limit):
    item = {"url": url, "path": str(path.relative_to(ROOT)), "bytes": 0,
            "status": "started"}
    record["assets"].append(item)
    save()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "Goal-Bus research asset fetch"})
    with urllib.request.urlopen(req, timeout=30) as response, part.open("xb") as f:
        item["http_status"] = response.status
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            item["bytes"] += len(block)
            if item["bytes"] > limit or time.monotonic() - started > 900:
                raise RuntimeError("Frozen asset budget exceeded")
            f.write(block)
            digest.update(block)
    item["sha256"] = digest.hexdigest()
    item["status"] = "downloaded"
    part.rename(path)
    save()
    return item

try:
    source = ROOT / "vendor/SafeDreamer"
    if source.exists():
        raise RuntimeError("Source target already exists; inspect before changing it")
    revision = cfg["source_revision"]
    archive = out / "source.tar.gz"
    download(f"https://codeload.github.com/PKU-Alignment/SafeDreamer/tar.gz/{revision}",
             archive, 100_000_000)
    with tarfile.open(archive) as tar:
        files = [m for m in tar.getmembers() if m.isfile()]
        if sum(m.size for m in files) > 200_000_000:
            raise RuntimeError("Unexpected source expansion size")
        for member in files:
            parts = Path(member.name).parts[1:]
            if not parts or any(p in ("..", "/") for p in parts):
                raise RuntimeError("Invalid archive member")
            target = source.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    record["source_files"] = len(files)
    record["source_revision"] = revision
    save()
    print(json.dumps({"event": "source_ready", "files": len(files)}), flush=True)
    ck = cfg["checkpoint"]
    target = ROOT / "models/SafeDreamer/osrp_pointgoal1_seed0.ckpt"
    if target.exists():
        raise RuntimeError("Checkpoint target already exists; do not repeat download")
    item = download(f"https://huggingface.co/{ck['repository']}/resolve/{ck['revision']}/{ck['file']}",
                    target, cfg["resource_limits"]["checkpoint_download_bytes_max"])
    if item["bytes"] != ck["bytes"] or item["sha256"] != ck["published_sha256"]:
        raise RuntimeError("Checkpoint differs from frozen official LFS metadata")
    item["verified_against_published_lfs"] = True
    record["status"] = "completed"
except Exception as exc:
    record["status"] = "failed"
    record["error_type"] = type(exc).__name__
    record["error"] = str(exc)
    raise
finally:
    save()
    print(json.dumps(record, indent=2), flush=True)
