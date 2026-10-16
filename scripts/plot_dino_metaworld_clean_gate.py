#!/usr/bin/env python3
"""Plot audited, real-environment Clean/no-op traces; never run a model or simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(record):
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"Changed source: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.analysis.read_text())
    if audit["status"] != "completed_independent_trace_audit":
        raise ValueError("A completed independent audit is required")
    summary_path = read_bound(audit["source"])
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed":
        raise ValueError("Cannot plot an unfinished panel")
    if [e["seed"] for e in summary["episodes"]] != [1, 2, 3]:
        raise ValueError("Expected the three frozen development seeds")
    if args.output.exists() or args.output.with_suffix(".sources.json").exists():
        raise FileExistsError("Use a fresh output; existing evidence is not overwritten")
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
        for ax, episode in zip(axes, summary["episodes"]):
            initial = episode["setup"]["initial_reach_distance"]
            for key, label, color in (("outcome", "Clean CEM", "#1867a5"),
                                      ("matched_noop", "Raw zero action", "#737373")):
                rows = [json.loads(line) for line in read_bound(episode[key]["trace"]).read_text().splitlines()]
                x = [1] + [r["environment_step"] for r in rows]
                y = [initial] + [r["reach_distance"] for r in rows]
                ax.plot(x, y, label=label, color=color, linewidth=1.6)
                ax.scatter(x[-1], y[-1], color=color, s=20, zorder=3)
            ax.axhline(.05, color="#b93532", linestyle="--", linewidth=1, label="Success threshold")
            success = episode["outcome"]["final_success"]
            ax.set_title(f"Seed {episode['seed']}: final {'success' if success else 'failure'}")
            ax.set_xlabel("Environment control step")
            ax.set_xlim(1, 102)
            ax.set_ylim(0, .32)
            ax.grid(alpha=.18)
        axes[0].set_ylabel("Actual TCP-to-target distance (m)")
        axes[1].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"DINO-WM / MetaWorld reach: Clean {audit['clean_successes']}/3, no-op {audit['noop_successes']}/3")
        fig.text(.5, .01, "Three fixed development seeds. Success is assessed only at step 100; not an attack or generalization test.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .94))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, metadata={"Date": None})
        plt.close(fig)
    provenance = {
        "analysis": {"path": str(args.analysis.resolve()), "sha256": digest(args.analysis)},
        "summary": audit["source"],
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "figure": {"path": str(args.output.resolve()), "sha256": digest(args.output)},
        "data": "Only archived real-environment distances; no interpolation or model predictions.",
        "model_calls": 0, "environment_steps": 0, "api_requests": 0,
    }
    args.output.with_suffix(".sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figure": str(args.output), "clean_successes": audit["clean_successes"]}))


if __name__ == "__main__":
    main()
