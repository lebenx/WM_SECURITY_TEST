"""Check whether redistributing uncertainty really preserves native forecasts.

Uses an existing learned MBRL-Lib ensemble; no retraining or planner changes.
Separates matching two moments from matching a distribution, and matching one
transition from matching the joint law of a trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from pilot_mbrl_coupling import make_model, model_hash


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/uncertainty_decomposition_v0"
SOURCE = ROOT / "runs/mbrl_coupling_cartpole_v0/model"


def save(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=False)
    data = np.load(SOURCE / "transitions.npz")
    indices = data["validation_indices"][:64]
    plan = {
        "selection": "First 64 stored validation transitions, with original recorded actions.",
        "indices": indices.tolist(),
        "model_source": str(SOURCE.relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((SOURCE / "model.pth").read_bytes()).hexdigest(),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "vendor/mbrl-lib"), "rev-parse", "HEAD"], text=True).strip(),
        "alpha_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "cdf_grid": "401 points per state/output coordinate, spanning all component means +/- 8 largest component standard deviations.",
        "gaussian_operation": "Shrink member means toward ensemble mean; add removed between-member marginal variance to each diagonal component variance.",
        "exact_mixture_operation": "q_i=(1-alpha)*p_i+alpha*mean_j(p_j), evaluated analytically as mixtures, not returned through the native Gaussian API.",
        "scope": "Numerical feasibility check, not a trained attack, task outcome, or novelty claim.",
        "training_updates": 0, "environment_steps": 0, "api_requests": 0, "gpu_use": False,
    }
    save("plan.json", plan)
    model = make_model()
    model.load(SOURCE)
    model.eval().requires_grad_(False)
    before = model_hash(model)
    obs, act = torch.from_numpy(data["obs"][indices]), torch.from_numpy(data["act"][indices])
    with torch.inference_mode():
        mu, logvar = model(model._get_model_input(obs, act), use_propagation=False)
    mu, var = mu.double(), logvar.double().exp()
    mean = mu.mean(0)
    residual = mu - mean
    between = torch.einsum("end,enf->ndf", residual, residual) / len(mu)
    within = torch.diag_embed(var.mean(0))
    covariance = between + within
    marginal_between = residual.square().mean(0)
    marginal_total = var.mean(0) + marginal_between
    lo = mu.min(0).values - 8 * var.sqrt().max(0).values
    hi = mu.max(0).values + 8 * var.sqrt().max(0).values
    grid = lo.unsqueeze(-1) + (hi-lo).unsqueeze(-1) * torch.linspace(0, 1, 401, dtype=torch.double)

    def component_cdf(means, variances):
        z = (grid.unsqueeze(0)-means.unsqueeze(-1))/variances.sqrt().unsqueeze(-1)
        return .5 * (1 + torch.erf(z / np.sqrt(2)))

    native_component_cdf = component_cdf(mu, var)
    native_cdf = native_component_cdf.mean(0)
    rows = []
    for alpha in plan["alpha_values"]:
        moved_mu = mean + (1-alpha)*residual
        moved_var = var + alpha*(2-alpha)*marginal_between
        moved_residual = moved_mu - moved_mu.mean(0)
        moved_covariance = (torch.einsum("end,enf->ndf", moved_residual, moved_residual)/len(mu)
                            + torch.diag_embed(moved_var.mean(0)))
        moved_cdf = component_cdf(moved_mu, moved_var).mean(0)
        per_coordinate_ks = (moved_cdf-native_cdf).abs().max(-1).values
        exact_nested = (1-alpha)*native_component_cdf + alpha*native_cdf.unsqueeze(0)
        row = {
            "alpha": alpha,
            "between_member_variance_fraction": (1-alpha)**2,
            "native_gaussian_family": {
                "mean_max_abs_error": float((moved_mu.mean(0)-mean).abs().max()),
                "diagonal_variance_max_abs_error": float((moved_var.mean(0)+moved_residual.square().mean(0)-marginal_total).abs().max()),
                "full_covariance_max_abs_error": float((moved_covariance-covariance).abs().max()),
                "coordinate_cdf_max_gap_on_grid": float(per_coordinate_ks.max()),
                "coordinate_cdf_mean_sup_gap_on_grid": float(per_coordinate_ks.mean()),
                "coordinates_above_1e_3": int((per_coordinate_ks>1e-3).sum()),
                "coordinates_evaluated": per_coordinate_ks.numel(),
            },
            "nested_mixture_family": {
                "cdf_max_abs_error_on_grid": float((exact_nested.mean(0)-native_cdf).abs().max()),
                "marginal_law_identity": "mean_i(q_i)=mean_i(p_i) algebraically for every outcome, not only this grid.",
                "native_interface_compatible": alpha == 0,
                "limitation": "For alpha>0 an outer member is generally a Gaussian mixture, whereas the native API returns one diagonal Gaussian per member.",
            },
        }
        assert row["native_gaussian_family"]["mean_max_abs_error"] < 1e-12
        assert row["native_gaussian_family"]["diagonal_variance_max_abs_error"] < 1e-12
        assert row["nested_mixture_family"]["cdf_max_abs_error_on_grid"] < 1e-12
        rows.append(row)
    assert rows[-1]["native_gaussian_family"]["coordinate_cdf_max_gap_on_grid"] > 1e-3
    # Exact two-step drift example: persistent model identity versus re-sampling.
    # A component index encodes an unknown fixed +1/-1 physical drift.
    fixed = {-2: .5, 0: 0., 2: .5}
    resampled = {-2: .25, 0: .5, 2: .25}
    temporal = {
        "one_step_both": {-1: .5, 1: .5},
        "two_step_persistent_identity": fixed,
        "two_step_resampled_identity": resampled,
        "two_step_total_variation": .5*sum(abs(fixed[k]-resampled[k]) for k in fixed),
        "meaning": "Identical one-step ensemble mixtures do not imply identical multi-step forecasts. This is the established PETS TS-infinity versus TS1 distinction, not a new mechanism.",
        "scope": "Constructed exact probability calculation, not native task execution.",
    }
    assert model_hash(model) == before
    np.savez_compressed(RUN / "native_predictions.npz", mean=mu.numpy(), variance=var.numpy(), indices=indices)
    result = {
        "status": "native_exact_forecast_preservation_gate_failed",
        "rows": rows, "temporal_counterexample": temporal,
        "interpretation": "Reducing ensemble disagreement while keeping mean and diagonal variance fixed changes the native learned predictive law. Exact repartition is possible by adding an inner mixture, but that changes the native output family; resampling that identity can also change trajectory laws. Generic uncertainty manipulation is prior work. No uncertainty-driven planner attack is justified by this screen.",
        "model_hash": before, "model_unchanged": True,
        "learned_ensemble_forward_calls": 1, "queried_transitions": len(indices),
        "training_updates": 0, "environment_steps": 0, "api_requests": 0, "gpu_use": False,
        "elapsed_s": time.monotonic()-start,
    }
    save("summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
