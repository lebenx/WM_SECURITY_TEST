"""One bounded retry of the same official Git blob; no checkpoint loading."""
import base64
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SHA = "1ff41a27b31864cf6cf9eafc5c7745cfd3e90444"
SIZE = 10026423
URL = "https://api.github.com/repos/tasl-lab/GenSafeNav/git/blobs/" + SHA
TARGET = ROOT / "vendor/GenSafeNav/trained_models/Ours_GST/checkpoints/05207.pt"
started = time.monotonic()
record = {"status": "downloading", "url": URL, "git_blob_sha1": SHA, "expected_bytes": SIZE,
          "hard_wall_seconds": 120, "transport_attempt": 1, "received_bytes": 0,
          "model_loads": 0, "environment_steps": 0}

def persist():
    record["elapsed_seconds"] = time.monotonic() - started
    (OUT / "actor_transport_retry.json").write_text(json.dumps(record, indent=2) + "\n")

def deadline(*_):
    raise TimeoutError("Official actor transport exceeded total120s wall budget")

signal.signal(signal.SIGALRM, deadline)
signal.setitimer(signal.ITIMER_REAL, 120)
persist()
try:
    req = urllib.request.Request(URL, headers={"Accept": "application/vnd.github.raw+json",
                                               "User-Agent": "GenSafeNav-pinned-blob-transport"})
    chunks = []
    with urllib.request.urlopen(req, timeout=30) as response:
        record["http_status"] = response.status
        record["content_type"] = response.headers.get("Content-Type")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            record["received_bytes"] += len(chunk)
            persist()
            print("Received", record["received_bytes"], "transport bytes", flush=True)
    data = b"".join(chunks)
    if "json" in (record["content_type"] or ""):
        blob = json.loads(data)
        assert blob["sha"] == SHA and blob["encoding"] == "base64"
        data = base64.b64decode(blob["content"])
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert len(data) == SIZE and actual == SHA, "Actor blob mismatch"
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".verified_api.tmp")
    temporary.write_bytes(data)
    temporary.replace(TARGET)
    record.update(status="completed_same_official_blob_verified", stored_bytes=len(data), git_blob_verified=True)
except BaseException as error:
    record.update(status="transport_failed", error={"type": type(error).__name__, "message": str(error)})
    raise
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    persist()
print(json.dumps(record))
