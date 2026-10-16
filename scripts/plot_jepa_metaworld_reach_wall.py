#!/usr/bin/env python3
"""Plot completed, independently audited reach-wall real-environment traces.

This script never imports a runner, model, or environment. It refuses unfinished
panels, changed source bindings, and existing output artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [1, 2, 3]
THRESHOLD = .05
CONDITIONS = (
    ("native", "Native JEPA-WM CEM", "#0072B2", "-"),
    ("wall_aware_servo", "Wall-aware P5", "#D55E00", "-"),
    ("straight_servo", "Straight P5", "#009E73", "-."),
    ("noop_repeat0", "Raw zero action", "#777777", ":"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def bound(record, sources):
    path = Path(record["path"]).resolve()
    actual = binding(path)
    require(actual["sha256"] == record["sha256"], f"Changed source: {path}")
    sources[str(path)] = actual["sha256"]
    return path


def read(record, sources):
    return json.loads(bound(record, sources).read_text())


def verify_bindings(value, sources):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            bound(value, sources)
        for child in value.values():
            verify_bindings(child, sources)
    elif isinstance(value, list):
        for child in value:
            verify_bindings(child, sources)


def near(left, right, label):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12), label)


def real_curve(episode, sources):
    require(episode["status"] == "completed", "Incomplete episode cannot be plotted")
    setup, outcome = episode["setup"], episode["outcome"]
    with np.load(bound(setup["setup"], sources), allow_pickle=False) as arrays:
        target, initial_tcp = arrays["target"].copy(), arrays["tcp"].copy()
    require(target.shape == initial_tcp.shape == (3,) and np.isfinite(target).all()
            and np.isfinite(initial_tcp).all(), "Invalid archived physical initial state")
    initial = float(np.linalg.norm(initial_tcp - target))
    near(initial, setup["initial_reach_distance"], "Initial physical distance differs")
    require(setup["elapsed_after_warmup"] == 1 and initial > THRESHOLD,
            "Unexpected initial observation or trivial initial state")
    rows = [json.loads(line) for line in bound(outcome["trace"], sources).read_text().splitlines()]
    require(len(rows) == 99 and [row["environment_step"] for row in rows] == list(range(2, 101)),
            "Incomplete or changed physical control budget")
    distances, successes = [], []
    for row in rows:
        tcp = np.asarray(row["tcp_after"], dtype=np.float64)
        require(tcp.shape == (3,) and np.isfinite(tcp).all(), "Invalid real TCP coordinate")
        distance = float(np.linalg.norm(tcp - target))
        near(distance, row["reach_distance"], "Saved distance differs from actual TCP/target")
        require(isinstance(row["success"], bool) and row["success"] == (distance <= THRESHOLD),
                "Native success differs from registered reach-wall threshold")
        require(row["physical"]["target"] == setup["arrays"]["target"], "Target changed during trajectory")
        distances.append(distance)
        successes.append(row["success"])
    near(distances[-1], outcome["final_reach_distance"], "Final distance differs from outcome")
    near(sum(row["reward"] for row in rows), outcome["reward_sum"], "Return differs from trace")
    require(outcome["final_success"] == successes[-1] and outcome["ever_success"] == any(successes)
            and outcome["total_environment_steps"] == 100 and outcome["policy_controls_executed"] == 99,
            "Outcome differs from complete real-environment trace")
    return {"x": [1] + list(range(2, 101)), "y": [initial] + distances,
            "final_success": successes[-1], "ever_success": any(successes),
            "reward_sum": outcome["reward_sum"], "trace": outcome["trace"],
            "initial_tcp": initial_tcp.tolist(), "target": target.tolist()}


def load_panel(analysis_path):
    sources = {}
    analysis_binding = binding(analysis_path)
    analysis = read(analysis_binding, sources)
    require(analysis["status"] == "completed_independent_reach_wall_native_audit",
            "A completed independent native audit is required")
    verify_bindings(analysis, sources)
    native = read(analysis["source"], sources)
    cpu = read(analysis["cpu_summary"], sources)
    config = read(analysis["config"], sources)
    cpu_audit = read(analysis["cpu_analysis"], sources)
    require(native["status"] == "completed" and cpu["status"] == "completed_cpu_validation"
            and cpu_audit["status"] == "completed_independent_reach_wall_cpu_audit",
            "Cannot plot an unfinished panel")
    require(native["config"] == analysis["config"] and native["cpu_summary"] == analysis["cpu_summary"]
            == config["cpu_summary"] == cpu_audit["source"], "Audit/source linkage differs")
    require(config["protocol"]["task"] == "mw-reach-wall" and config["seeds"] == SEEDS
            and analysis["denominator_per_condition"] == 3, "Unexpected task or panel")
    for record in (native, cpu, analysis):
        require([episode["seed"] for episode in record["episodes"]] == SEEDS,
                "The three fixed seeds must be complete and ordered")
    panel = []
    counts = {condition: 0 for condition, *_ in CONDITIONS}
    for native_episode, cpu_episode, row in zip(native["episodes"], cpu["episodes"], analysis["episodes"]):
        require(cpu_episode["status"] == "completed", "Incomplete CPU episode")
        branches = {branch["condition"]: branch for branch in cpu_episode["branches"]}
        require(list(branches) == ["noop_repeat0", "noop_repeat1", "straight_servo", "wall_aware_servo"],
                "CPU branch panel changed")
        curves = {"native": real_curve(native_episode, sources)}
        for condition, *_ in CONDITIONS[1:]:
            branch = branches[condition]
            require(native_episode["setup"]["arrays"] == branch["setup"]["arrays"],
                    "Initial state/goal pairing differs")
            curves[condition] = real_curve(branch, sources)
        for condition, curve in curves.items():
            require(row[condition + "_final_success"] == curve["final_success"]
                    and row[condition + "_ever_success"] == curve["ever_success"],
                    "Audit success differs from physical trace")
            near(row[condition + "_final_reach_distance"], curve["y"][-1], "Audit final distance differs")
            return_key = "native_policy_return" if condition == "native" else condition + "_reward_sum"
            near(row[return_key], curve["reward_sum"], "Audit return differs")
            counts[condition] += int(curve["final_success"])
        panel.append({"seed": native_episode["seed"], "curves": curves})
    require(counts["native"] == analysis["native_final_successes"]
            and {c: n for c, n in counts.items() if c != "native"} == analysis["cpu_final_successes"],
            "Audit aggregate success differs")
    require(analysis["gate_passed"] == native["gate"]["passed"], "Saved gate differs")
    return analysis_binding, analysis, panel, counts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=Path("runs/jepa_metaworld_reach_wall_native_analysis_v1/analysis.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(".sources.json")
    require(output.suffix.lower() == ".svg", "Output must be an SVG")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Use fresh output paths; existing evidence is not overwritten")
    analysis_binding, analysis, panel, counts, sources = load_panel(args.analysis)
    maximum = max(max(curve["y"]) for seed in panel for curve in seed["curves"].values())
    # Use one shared scale covering every actual observation; never crop failures.
    ymax = max(.32, math.ceil(maximum * 1.08 / .05) * .05)
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "jepa-reach-wall-v1"}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        for ax, episode in zip(axes, panel):
            for condition, label, color, linestyle in CONDITIONS:
                curve = episode["curves"][condition]
                success = curve["final_success"]
                ax.plot(curve["x"], curve["y"], color=color, linestyle=linestyle,
                        linewidth=2 if condition == "native" else 1.5,
                        label=label + (": success" if success else ": failure"))
                ax.scatter(100, curve["y"][-1], color=color, marker="o" if success else "x",
                           s=36, linewidths=1.6, zorder=5)
            ax.axhline(THRESHOLD, color="#9B2335", linestyle="--", linewidth=1,
                       label="Final success threshold: 0.05 m")
            ax.set_title(f"Seed {episode['seed']}")
            ax.set_xlabel("Environment control step")
            ax.set_xlim(1, 103)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=.18)
            ax.legend(loc="upper right", fontsize=7.8, framealpha=.94)
        axes[0].set_ylabel("Actual TCP-to-target distance (m)")
        fig.suptitle("JEPA-WM / MetaWorld reach-wall: actual environment trajectories", y=.98)
        fig.text(.5, .90,
                 f"Final success: native {counts['native']}/3 | wall-aware P5 {counts['wall_aware_servo']}/3"
                 f" | straight P5 {counts['straight_servo']}/3 | no-op {counts['noop_repeat0']}/3",
                 ha="center", fontsize=10)
        fig.text(.5, .045, "Endpoint markers: circle = success, cross = failure at step 100. All 99 policy controls are shown.",
                 ha="center", fontsize=9)
        fig.text(.5, .012, "Three fixed development seeds; no-op repeats are not extra samples. Normal capability screen, not an attack test.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .085, 1, .89))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            fig.savefig(handle, format="svg", metadata={"Date": None})
        plt.close(fig)
    provenance = {
        "status": "plotted_completed_independent_native_audit", "analysis": analysis_binding,
        "native_summary": analysis["source"], "cpu_summary": analysis["cpu_summary"],
        "script": binding(__file__), "figure": binding(output),
        "verified_source_sha256": sources,
        "final_success_counts": counts, "denominator_per_condition": 3,
        "native_gate_passed": analysis["gate_passed"],
        "plotted_control_count": 3 * 4 * 99,
        "curves": [{"seed": episode["seed"], "condition": condition,
                    "trace": curve["trace"], "final_success": curve["final_success"],
                    "final_reach_distance": curve["y"][-1]}
                   for episode in panel for condition, curve in episode["curves"].items()],
        "data": "Archived real TCP coordinates minus the fixed actual target; no forecast or CEM objective is used. Straight segments connect consecutive recorded controls; no smoothing or resampling.",
        "scope": "Three development seeds, not a population or security conclusion. Wall-aware P5 uses public wall geometry and matched feedback times; native CEM uses more computation and can vary controls within each chunk.",
        "model_calls": 0, "environment_steps": 0, "api_requests": 0,
    }
    with sidecar.open("x") as handle:
        handle.write(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figure": str(output), "sources": str(sidecar), "final_success_counts": counts}))


if __name__ == "__main__":
    main()
