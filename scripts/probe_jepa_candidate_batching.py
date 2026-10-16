#!/usr/bin/env python3
"""Paired fixed-real-observation CPU native300 vs microbatch50 forecast audit.

No environment simulation, task outcome, attack, training or CUDA invocation.
Candidate sampling happens once before both paths and global top10 statistics
are compared afterwards. This check does not establish full GPU300 parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import pilot_jepa_native_task_gate as native
from jepa_candidate_batching import install_candidate_microbatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=native.ROOT / "runs/jepa_candidate_batching_cpu_v0")
    parser.add_argument("--source-dir", type=Path, default=native.ROOT / "runs/jepa_native_task_gate_cpu_v1")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    native.configure_process("cpu")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Preserve existing evidence: output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"status": "preparing", "device": "cpu", "source_dir": str(args.source_dir.resolve()),
              "native_candidates": 300, "microbatch_size": 50, "horizon": 6, "seed": 1,
              "full_gpu300_parity_established": False, "task_outcomes_measured": False,
              "scope": "One fixed candidate set on actual cached initial/goal observations. No claim about all CEM iterations or GPU full-batch numerical equality.",
              "sha256": {str(p): native.sha(p) for p in (
                  Path(__file__), native.ROOT / "scripts/jepa_candidate_batching.py",
                  native.ROOT / "scripts/pilot_jepa_native_task_gate.py",
                  args.source_dir / "initial_goal.npz", args.source_dir / "protocol.json")}}
    native.write_json(out / "summary.json", result)
    try:
        import numpy as np
        import torch
        import yaml
        from omegaconf import OmegaConf
        from tensordict import TensorDict
        sys.path.insert(0, str(native.UPSTREAM))
        torch.set_num_threads(args.threads)
        original = yaml.safe_load(native.CONFIG.read_text())
        cfg = OmegaConf.create(original)
        cfg.work_dir = out
        cfg.device = "cpu"
        cfg.local_seed = 1
        cfg.planner.decode_each_iteration = False
        model, pre, stats, load_checks = native.load_model(original)
        agent = native.build_agent(cfg, model, pre)
        source = np.load(args.source_dir / "initial_goal.npz")
        goal = TensorDict({"visual": torch.from_numpy(source["goal_pixels"]),
                           "proprio": torch.from_numpy(source["goal_state"]).unsqueeze(0)})
        initial = TensorDict({"visual": torch.from_numpy(source["initial_pixels"]),
                              "proprio": torch.from_numpy(source["initial_state"]).unsqueeze(0)})
        agent.set_goal(goal)
        z = model.encode(initial.unsqueeze(0), act=True)
        generator = torch.Generator(device="cpu").manual_seed(1)
        candidates = torch.randn(6, 300, model.action_dim, generator=generator)
        candidates[:, 0] = 0  # Native first-iteration mean inclusion.
        candidate_before = candidates.clone()
        torch.save({"normalized_candidates": candidates, "encoded_initial": z,
                    "encoded_goal": agent.goal_state_enc}, out / "inputs.pt")
        costs, elapsed, states = {}, {}, {}
        with torch.no_grad():
            for label in ("native_full300", "microbatch50"):
                if label == "microbatch50":
                    install_candidate_microbatch(agent.planner, 50)
                rng_before = torch.get_rng_state().clone()
                tick = time.monotonic()
                costs[label] = agent.planner.cost_function(candidates, z).clone()
                elapsed[label] = time.monotonic()-tick
                states[label] = torch.equal(rng_before, torch.get_rng_state())
                torch.save(costs[label], out / f"{label}_costs.pt")
                assert costs[label].shape == (300,) and torch.isfinite(costs[label]).all()
                assert torch.equal(candidates, candidate_before)
                result.update(status=label+"_completed", elapsed_per_mode=elapsed,
                              rng_unchanged=states)
                native.write_json(out / "summary.json", result)
                print(json.dumps({"mode": label, "elapsed_seconds": elapsed[label]}), flush=True)
        full, micro = costs.values()
        full_idx = torch.topk(-full, 10).indices
        micro_idx = torch.topk(-micro, 10).indices
        full_actions, micro_actions = candidates[:, full_idx], candidates[:, micro_idx]
        full_mean, micro_mean = full_actions.mean(1), micro_actions.mean(1)
        full_std, micro_std = full_actions.std(1), micro_actions.std(1)
        full_sorted = torch.sort(full).values
        absolute = (full-micro).abs()
        result.update(status="completed", checkpoint_load_checks=load_checks,
                      costs_exact_equal=bool(torch.equal(full,micro)),
                      costs_allclose_atol1e6_rtol1e5=bool(torch.allclose(full,micro,atol=1e-6,rtol=1e-5)),
                      max_abs_cost_difference=float(absolute.max()),
                      max_rel_cost_difference=float((absolute/full.abs().clamp_min(1e-12)).max()),
                      native_top10_indices=full_idx.tolist(), microbatch_top10_indices=micro_idx.tolist(),
                      elite_order_exact_equal=bool(torch.equal(full_idx,micro_idx)),
                      elite_set_equal=bool(torch.equal(full_idx.sort().values,micro_idx.sort().values)),
                      next_CEM_mean_exact_equal=bool(torch.equal(full_mean,micro_mean)),
                      next_CEM_std_exact_equal=bool(torch.equal(full_std,micro_std)),
                      max_abs_next_mean_difference=float((full_mean-micro_mean).abs().max()),
                      max_abs_next_std_difference=float((full_std-micro_std).abs().max()),
                      native_elite_boundary_cost_margin=float(full_sorted[10]-full_sorted[9]),
                      cuda_initialized=torch.cuda.is_initialized())
        native.write_json(out / "cost_vectors.json", {k:v.tolist() for k,v in costs.items()})
        torch.save({"native_elites":full_idx,"micro_elites":micro_idx,
                    "native_mean":full_mean,"micro_mean":micro_mean,
                    "native_std":full_std,"micro_std":micro_std}, out / "CEM_statistics.pt")
        assert not torch.cuda.is_initialized()
    except BaseException as exc:
        result.update(status="runtime_error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        result["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        native.write_json(out / "summary.json", result)


if __name__ == "__main__":
    main()
