"""Bounded CPU check of the unmodified DiscoverPhysics experiment interface.

These are researcher-selected, noise-free fixtures, not learned experiments or
benchmark discovery results. Hidden law descriptions never enter the records.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/DiscoverPhysics"
sys.path.insert(0, str(VENDOR / "ScienceAgent"))


def main():
    import numpy as np
    from scienceagent.worlds import get_world

    output = ROOT / "runs/discoverphysics_interface_prepare_v1"
    output.mkdir(exist_ok=False)
    save = lambda name, value: (output / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    started = time.monotonic()
    records = []
    for world in ["gravity", "coulomb_easy"]:
        executor = get_world(world, engine="nbody", noise_std=0.0)["executor"]
        for p2 in [1.0, 2.0]:
            request = dict(p1=1.0, p2=p2, pos2=[5.0, 0.0],
                           velocity2=[0.0, 0.0],
                           measurement_times=[0.0, 0.025, 0.05, 0.1, 0.2],
                           duration=5.0)
            result = executor.run([request])[0]
            for key in ["pos2", "velocity2"]:
                arr = np.asarray(result[key])
                assert arr.shape == (5, 2), (world, key, arr.shape)
                assert np.isfinite(arr).all(), (world, key)
            records.append(dict(world=world, request=request, response=result))
            save("requests_responses.json", records)
            print(json.dumps(dict(world=world, p2=p2,
                                  completed_native_experiments=len(records))), flush=True)
    summary = dict(
        status="completed", native_experiments=len(records), fixtures=2,
        api_requests=0, gpu_work=False, engine="nbody", noise_std=0.0,
        pin=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=VENDOR, text=True).strip(),
        versions={p: importlib.metadata.version(p) for p in ["jax", "numpy", "scipy"]},
        elapsed_seconds=time.monotonic() - started,
        scope="Interface preparation only: researcher-supplied requests, no learned "
              "experiment compiler, transfer advantage, or discovery task score.",
        prior_attempt="v0 failed at package import (missing requests); no native experiment ran.",
    )
    save("summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
