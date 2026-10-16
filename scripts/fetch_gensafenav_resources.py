#!/usr/bin/env python3
"""Fetch a pinned, minimal GenSafeNav runtime and two matching official weights.

No model deserialization, model inference, environment construction or training.
Existing files must match their official Git blob IDs; mismatches fail closed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "01baf926a5c77c1a4ab28635658eb014ef4f1767"
REPOSITORY = "https://github.com/tasl-lab/GenSafeNav"
GST = "gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj/checkpoint"
REQUIRED = {
    "trained_models/Ours_GST/checkpoints/05207.pt": (10026423, "1ff41a27b31864cf6cf9eafc5c7745cfd3e90444"),
    GST + "/epoch_100.pt": (827916, "0b45db5ccaf04068e87e742af33115d1db61045a"),
    GST + "/args.pickle": (746, "1b18d934750f82845aa8307d2c6603be3194b0c2"),
}


def request(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "GenSafeNav-pinned-resource-fetch"}), timeout=60) as response:
        return response.read()


def blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def selected(path):
    if path in REQUIRED or path in {"README.md", "LICENSE", "Dockerfile", "test.py", "arguments.py"}:
        return True
    if path.startswith(("crowd_nav/", "crowd_sim/", "rl/", "baselines/", "dt_aci/", "configs/",
                        "gst_updated/src/", "gst_updated/scripts/wrapper/", "trained_models/Ours_GST/")):
        return path.endswith(".py")
    if path.startswith("Python-RVO2/") and not any(p in path.split("/") for p in ("build", "dist", "pyrvo2.egg-info", "doc")):
        # Cython generates rvo2.cpp locally in a separate build copy.
        return (path.endswith((".h", ".pyx", ".cmake")) or path.endswith("CMakeLists.txt")
                or (path.endswith(".cpp") and not path.endswith("/rvo2.cpp"))
                or path in {"Python-RVO2/setup.py", "Python-RVO2/LICENSE", "Python-RVO2/README.md"})
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "vendor/GenSafeNav")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gensafenav_resources_20261015")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tree_path = args.output / "official_tree.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
    else:
        raw = request(f"https://api.github.com/repos/tasl-lab/GenSafeNav/git/trees/{REVISION}?recursive=1")
        tree = json.loads(raw)
        tree_path.write_bytes(raw)
    assert tree["sha"] == REVISION and not tree["truncated"]
    entries = [e for e in tree["tree"] if e["type"] == "blob" and selected(e["path"])]
    by_path = {e["path"]: e for e in entries}
    for path, (size, sha) in REQUIRED.items():
        assert by_path[path]["size"] == size and by_path[path]["sha"] == sha
    args.destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "fetching", "repository": REPOSITORY, "revision": REVISION,
              "destination": str(args.destination), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "Runtime Python packages and clean RVO2 build sources; only Ours_GST actor05207 and GST _rand epoch100 plus its args.pickle. No nonrand or other actor weights.",
              "files": [], "models_loaded": 0, "model_forward_calls": 0, "environment_steps": 0}
    report_path = args.output / "provenance.json"

    def fetch(entry):
        path = entry["path"]
        target = args.destination / path
        url = f"https://raw.githubusercontent.com/tasl-lab/GenSafeNav/{REVISION}/{path}"
        local = ROOT / "runs/gensafenav_feasibility_20261015/sources" / path
        if target.exists():
            data, method = target.read_bytes(), "verified_existing"
        elif local.exists():
            data, method = local.read_bytes(), "verified_existing_source_snapshot"
        else:
            data, method = request(url), "official_raw_download"
        if len(data) != entry["size"] or blob_id(data) != entry["sha"]:
            raise RuntimeError(f"Git blob/size mismatch: {path}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return {"path": path, "bytes": len(data), "git_blob_sha1": entry["sha"], "git_blob_verified": True,
                "url": url, "method": method}

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = [pool.submit(fetch, entry) for entry in entries]
            for future in as_completed(pending):
                report["files"].append(future.result())
                if len(report["files"]) % 25 == 0:
                    print(f"Verified {len(report['files'])}/{len(entries)} selected files", flush=True)
        report["files"].sort(key=lambda entry: entry["path"])
        report.update(status="completed_git_blob_verified_resource_fetch", file_count=len(entries),
                      total_bytes=sum(entry["bytes"] for entry in report["files"]),
                      required_weight_and_args_paths=list(REQUIRED))
    except BaseException as error:
        report.update(status="resource_fetch_incomplete", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "files": report["file_count"], "bytes": report["total_bytes"]}))


if __name__ == "__main__":
    main()
