#!/usr/bin/env python3
"""Native JEPA-WM CEM with explicitly recorded candidate-forward microbatches.

All 300 candidates are sampled once per iteration; global top10/mean/std and
all 30 CEM iterations stay upstream. A first-call full25 versus13+12 diagnostic
does not assert parity to native GPU300. Default case is the first official
rank0 episode initially outside success: index2/seed3, selected by physics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pilot_jepa_native_task_gate as native
from jepa_candidate_batching import install_candidate_microbatch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-microbatch-size", type=int, default=25)
    own, forwarded = parser.parse_known_args()
    if own.candidate_microbatch_size != 25:
        raise ValueError("This prospective pilot fixes candidate microbatch size25")
    defaults = {"--device": "cuda", "--episode-index": "2", "--cuda-memory-limit-gib": "3",
                "--out-dir": str(native.ROOT / "runs/jepa_native_microbatch_ep2_v1")}
    for key, value in defaults.items():
        if key not in forwarded:
            forwarded += [key, value]
    device = forwarded[forwarded.index("--device")+1]
    episode_index = int(forwarded[forwarded.index("--episode-index")+1])
    if episode_index != 2:
        raise ValueError("This preregistered task gate fixes official episode index2/seed3")
    out = Path(forwarded[forwarded.index("--out-dir")+1]).resolve()
    audit_path = native.ROOT / "runs/jepa_candidate_batching_cpu_v0/summary.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "completed" or not all(audit.get(k) is True for k in (
        "costs_allclose_atol1e6_rtol1e5", "elite_order_exact_equal",
        "next_CEM_mean_exact_equal", "next_CEM_std_exact_equal")):
        raise RuntimeError("Required fixed-real-observation CPU batching gate not passed")
    for source in (native.ROOT / "scripts/pilot_jepa_native_task_gate.py",
                   native.ROOT / "scripts/jepa_candidate_batching.py"):
        if audit["sha256"][str(source)] != native.sha(source):
            raise RuntimeError("Execution adapter source changed since CPU gate")
    registration_path = native.ROOT / "configs/jepa_native_microbatch_ep2_v1.json"
    registration = json.loads(registration_path.read_text())
    for source, expected in registration["source_sha256"].items():
        if native.sha(native.ROOT / source) != expected:
            raise RuntimeError("Prospective source manifest changed")
    native.configure_process(device)  # Reexec before imports, preserving original CLI.
    sys.argv = [sys.argv[0], *forwarded]
    metadata = {
        "mode": "candidate_forward_microbatch25_native_CEM300x30",
        "candidate_sampling": "One native complete [6,300,10] draw per CEM iteration; unchanged local generator/seed; native mean sample inclusion.",
        "global_reduction": "Concatenate 300 scores in original candidate order before unchanged native top10, mean and std.",
        "microbatch_size": 25,
        "cpu_batching_gate_summary": str(audit_path),
        "cpu_batching_gate_sha256": native.sha(audit_path),
        "prospective_protocol": str(registration_path),
        "prospective_protocol_sha256": native.sha(registration_path),
        "native_GPU300_numerical_equivalence_established": False,
        "diagnostic": "First task candidate set compares full25 scores with13+12 and records all vectors, RNG and elite differences. No claim about all candidates/all iterations.",
        "case_selection": "Official rank0 first nontrivial initial task by XY distance>=0.5, checked before task outcomes: ep0/ep1 initially successful; ep2/seed3 initial distance2.488. No successful-task filtering.",
        "source_sha256": {str(p.relative_to(native.ROOT)): native.sha(p) for p in (
            Path(__file__), native.ROOT / "scripts/jepa_candidate_batching.py",
            native.ROOT / "scripts/pilot_jepa_native_task_gate.py")},
    }
    original_write = native.write_json

    def write_json(path, value):
        if Path(path).name in ("summary.json", "protocol.json"):
            value = dict(value, execution_adapter=metadata)
            if Path(path).name == "protocol.json":
                value["scope"] = "One metadata-selected nontrivial native PointMaze episode; explicit microbatch execution adaptation, no success-rate/attack claim."
        original_write(path, value)

    original_build = native.build_agent

    def build_agent(cfg, model, pre):
        import torch
        agent = original_build(cfg, model, pre)
        full_cost = install_candidate_microbatch(agent.planner, 25)
        micro_cost = agent.planner.cost_function
        checked = False

        def cost(actions, z_init):
            nonlocal checked
            rng_before = torch.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state().clone() if device == "cuda" else None
            actual = micro_cost(actions, z_init)
            if not checked:
                full25 = actual[:25]
                micro25 = torch.cat([full_cost(actions[:, :13], z_init),
                                     full_cost(actions[:, 13:25], z_init)], dim=0)
                torch.save({"normalized_candidates25": actions[:, :25].detach().cpu(),
                            "full25_costs": full25.detach().cpu(), "micro13plus12_costs": micro25.detach().cpu()},
                           out / "first_candidates25_batch_diagnostic.pt")
                cpu_same = torch.equal(rng_before, torch.get_rng_state())
                cuda_same = device != "cuda" or torch.equal(cuda_before, torch.cuda.get_rng_state())
                diagnostic = {"status": "completed", "device": device, "full_candidates": 25,
                    "diagnostic_slices": [13, 12], "costs_exact_equal": bool(torch.equal(full25,micro25)),
                    "max_abs_cost_difference": float((full25-micro25).abs().max()),
                    "costs_allclose_atol1e6_rtol1e5": bool(torch.allclose(full25,micro25,atol=1e-6,rtol=1e-5)),
                    "full25_top10": torch.topk(-full25,10).indices.tolist(),
                    "micro25_top10": torch.topk(-micro25,10).indices.tolist(),
                    "cpu_rng_unchanged": cpu_same, "cuda_rng_unchanged": cuda_same,
                    "full_GPU300_parity_established": False,
                    "scope": "Only first25 real task candidates in firstCEM iteration; complete task uses microbatch25 throughout."}
                original_write(out / "first_candidates25_batch_diagnostic.json", diagnostic)
                if not cpu_same or not cuda_same:
                    raise RuntimeError("Forecast batching/diagnostic consumed RNG")
                if not diagnostic["costs_allclose_atol1e6_rtol1e5"]:
                    raise RuntimeError("Large score discrepancy in fixed-candidate execution diagnostic")
                checked = True
            return actual

        agent.planner.cost_function = cost
        return agent

    native.write_json = write_json
    native.build_agent = build_agent
    native.main()


if __name__ == "__main__":
    main()
