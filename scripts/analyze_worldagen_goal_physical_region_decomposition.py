#!/usr/bin/env python3
"""Decompose Goal-swap forecast error on physical-motion vs static patch support.

The mask is derived from the same fixed replay target and the current processed
image: patches whose normalized target differs from the current observation are
treated as a motion-support proxy. This is a post-hoc target analysis, not a
new planner or attack result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TASKS = (0, 2, 4, 6, 7, 9)
SEEDS = (17, 31, 37)


def patchify(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = 16
    z = x.reshape(n, c, h // p, p, w // p, p)
    z = torch.einsum("nchpwq->nhwpqc", z).reshape(n, (h // p) * (w // p), p * p * c)
    return (z - z.mean(-1, keepdim=True)) / (z.var(-1, keepdim=True) + 1e-6).sqrt()


def mean_ci(records: list[dict], key: str, clusters: int = 18,
            draws: int = 20000, seed: int = 20261012) -> dict:
    vals = np.asarray([float(r[key]) for r in records], dtype=float)
    cluster_ids = np.asarray([int(r["cluster_id"]) for r in records])
    unique = np.unique(cluster_ids)
    means = np.asarray([
        vals[cluster_ids == c].mean() for c in unique
    ], dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(means), size=(draws, len(means)))
    boot = means[sample].mean(axis=1)
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "task_cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "positive_fraction": float(np.mean(vals > 0)),
        "cluster_count": int(len(unique)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "runs/worldagen_aligned_goal_sweep_h30_all6_seed17",
        "runs/worldagen_aligned_goal_sweep_h30_all6_seed31",
        "runs/worldagen_aligned_goal_sweep_h30_all6_seed37",
    ])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    records: list[dict] = []
    cluster = 0
    for root_name in args.roots:
        root = Path(root_name)
        summary = json.loads((root / "summary.json").read_text())
        seed = int(summary["seed"])
        for row in summary["rows"]:
            task = int(row["task_id"])
            forecasts = torch.load(root / f"task_{task}_forecasts.pt", map_location="cpu", weights_only=False)
            targets = torch.load(root / f"task_{task}_targets.pt", map_location="cpu", weights_only=False)["target_patches_step3"].float()
            current = patchify(forecasts["reference_processed"].unsqueeze(0))[0]
            motion_score = (targets - current).square().mean(-1)[0].numpy()
            dynamic = motion_score > 1e-8
            static = ~dynamic
            masks = {"dynamic_support": dynamic, "static_support": static, "all": np.ones_like(dynamic, dtype=bool)}
            clean = forecasts["worlds_step3"]["clean"].float()
            for offset in range(1, 10):
                swap = forecasts["worlds_step3"][f"swap_{offset}"].float()
                bus = forecasts["worlds_step3"][f"swap_{offset}_bus_cut"].float()
                control = forecasts["worlds_step3"][f"swap_{offset}_control_cut"].float()
                for candidate in range(targets.shape[0]):
                    for region, mask in masks.items():
                        if not np.any(mask):
                            continue
                        clean_err = ((clean[candidate] - targets[candidate]).square().mean(-1).numpy()[mask]).mean()
                        values = {
                            "swap": ((swap[candidate] - targets[candidate]).square().mean(-1).numpy()[mask]).mean(),
                            "bus_cut": ((bus[candidate] - targets[candidate]).square().mean(-1).numpy()[mask]).mean(),
                            "control_cut": ((control[candidate] - targets[candidate]).square().mean(-1).numpy()[mask]).mean(),
                        }
                        records.append({
                            "seed": seed, "task_id": task, "cluster_id": cluster,
                            "offset": offset, "candidate": candidate, "region": region,
                            "clean_error": float(clean_err),
                            "swap_minus_clean": float(values["swap"] - clean_err),
                            "bus_cut_minus_clean": float(values["bus_cut"] - clean_err),
                            "control_cut_minus_clean": float(values["control_cut"] - clean_err),
                            "motion_patch_fraction": float(np.mean(dynamic)),
                        })
            cluster += 1

    by_region_offset: dict[str, dict[str, dict]] = {}
    for region in ("dynamic_support", "static_support", "all"):
        by_region_offset[region] = {}
        for offset in range(1, 10):
            subset = [r for r in records if r["region"] == region and r["offset"] == offset]
            by_region_offset[region][str(offset)] = {
                "swap_minus_clean": mean_ci(subset, "swap_minus_clean"),
                "bus_cut_minus_clean": mean_ci(subset, "bus_cut_minus_clean"),
                "control_cut_minus_clean": mean_ci(subset, "control_cut_minus_clean"),
            }
    pooled = {}
    for region in ("dynamic_support", "static_support", "all"):
        subset = [r for r in records if r["region"] == region]
        pooled[region] = {
            "swap_minus_clean": mean_ci(subset, "swap_minus_clean"),
            "bus_cut_minus_clean": mean_ci(subset, "bus_cut_minus_clean"),
            "control_cut_minus_clean": mean_ci(subset, "control_cut_minus_clean"),
            "mean_motion_patch_fraction": float(np.mean([r["motion_patch_fraction"] for r in subset])),
        }
    result = {
        "status": "completed_goal_physical_region_decomposition",
        "source_roots": [str(x) for x in args.roots],
        "tasks": list(TASKS), "seeds": list(SEEDS),
        "task_seed_pairs": 18, "counterfactuals": 162,
        "candidate_records": 144 * 9,
        "mask_definition": "dynamic_support is target-vs-current normalized patch MSE > 1e-8; static_support is its complement",
        "estimand": "forecast patch MSE change relative to Clean on a fixed physical replay target",
        "pooled": pooled,
        "by_region_offset": by_region_offset,
        "records": records,
        "scope_limits": [
            "Motion support is a target-derived spatial proxy, not a simulator object segmentation mask.",
            "This is a post-hoc forecast-error decomposition; it does not add a new selected-action utility estimate.",
            "The fixed target is the decoded primary-view patch after three executed actions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
