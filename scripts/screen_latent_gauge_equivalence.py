"""CPU algebra screen using LeWM's native criterion, not a trained-model attack.

Checks coordinate conjugacy, cost transport, and the linear-only scope of
covariance whitening. Fixed constructed candidates are not task experiments.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/latent_gauge_equivalence_v0"
UPSTREAM = ROOT / "vendor/le-wm/jepa.py"


def twist(x, inverse=False):
    # Radius is invariant, so changing the angle's sign is the exact inverse.
    angle = np.pi / (1 - 0.9**2) * np.square(x).sum(axis=-1)
    if inverse:
        angle = -angle
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*x[..., 0]-s*x[..., 1],
                     s*x[..., 0]+c*x[..., 1]), axis=-1)


def mahalanobis(candidates, goal, bank):
    delta = candidates-goal
    inverse_cov = np.linalg.inv(np.cov(bank, rowvar=False))
    return np.einsum("ni,ij,nj->n", delta, inverse_cov, delta)


def main():
    start = time.monotonic()
    RUN.mkdir(parents=True, exist_ok=False)
    plan = {
        "scope": "Fixed constructed states and exact additive dynamics; only the LeWM criterion is native. No pretrained checkpoint, simulator, or task harm.",
        "seed": 20261009,
        "linear_map": [[10.0, 0.0], [0.0, 0.1]],
        "twist_angle": "pi/(1-0.9**2) * squared_radius",
        "reference_bank_size": 20000,
        "conjugacy_trajectories": 128,
        "conjugacy_horizon": 5,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM.parent), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_jepa_sha256": hashlib.sha256(UPSTREAM.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_updates": 0, "gpu_use": False, "api_requests": 0,
    }
    (RUN/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    spec = importlib.util.spec_from_file_location("screen_lewm_jepa", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.JEPA(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity())

    def cost(candidates, goal):
        # The native criterion expects B, samples, time, latent dimension.
        return model.criterion({
            "predicted_emb": torch.tensor(candidates, dtype=torch.float64)[None, :, None, :],
            "goal_emb": torch.tensor(goal, dtype=torch.float64)[None, None, None, :],
        }).numpy()[0]

    rng = np.random.default_rng(plan["seed"])
    bank = rng.normal(size=(plan["reference_bank_size"], 2))
    matrix = np.asarray(plan["linear_map"])
    candidates = np.array([[1., 0.], [2., 2.]])
    goal = np.array([2., 0.])
    clean = cost(candidates, goal)
    changed = cost(candidates@matrix.T, goal@matrix.T)
    direct_metric = np.einsum("ni,ij,nj->n", candidates-goal,
                              matrix.T@matrix, candidates-goal)
    white_clean = mahalanobis(candidates, goal, bank)
    white_changed = mahalanobis(candidates@matrix.T, goal@matrix.T, bank@matrix.T)
    linear = {
        "clean_costs": clean.tolist(), "transformed_costs": changed.tolist(),
        "direct_metric_costs": direct_metric.tolist(),
        "clean_whitened_costs": white_clean.tolist(),
        "transformed_whitened_costs": white_changed.tolist(),
        "metric_equivalence_max_error": float(np.max(np.abs(changed-direct_metric))),
        "whitening_equivalence_max_error": float(np.max(np.abs(white_clean-white_changed))),
    }

    goal = np.array([1., 0.])
    angle = np.pi/(1-0.9**2)*(1-2**2)
    candidates = np.array([[0.9, 0.], [2*np.cos(angle), 2*np.sin(angle)]])
    changed_bank = twist(bank)
    changed_candidates, changed_goal = twist(candidates), twist(goal)
    clean = cost(candidates, goal)
    changed = cost(changed_candidates, changed_goal)
    transported = cost(twist(changed_candidates, True), twist(changed_goal, True))
    white_clean = mahalanobis(candidates, goal, bank)
    white_changed = mahalanobis(changed_candidates, changed_goal, changed_bank)
    z = rng.normal(size=(128, 2))
    original = z.copy()
    y = twist(z)
    path_errors = []
    for action in rng.normal(scale=.1, size=(5, 128, 2)):
        z += action
        y = twist(twist(y, True)+action)
        path_errors.append(float(np.max(np.abs(twist(y, True)-z))))
    predicted, actual = original+.01, original
    nonlinear = {
        "goal": goal.tolist(), "candidates": candidates.tolist(),
        "clean_costs": clean.tolist(), "transformed_costs": changed.tolist(),
        "transported_objective_costs": transported.tolist(),
        "clean_whitened_costs": white_clean.tolist(),
        "transformed_whitened_costs": white_changed.tolist(),
        "clean_selection": int(clean.argmin()),
        "transformed_selection": int(changed.argmin()),
        "transported_selection": int(transported.argmin()),
        "clean_whitened_selection": int(white_clean.argmin()),
        "transformed_whitened_selection": int(white_changed.argmin()),
        "norm_preservation_max_error": float(np.max(np.abs(np.linalg.norm(bank, axis=1)-np.linalg.norm(changed_bank, axis=1)))),
        "conjugacy_decoded_max_errors_per_step": path_errors,
        "original_nonzero_prediction_mse": float(np.square(predicted-actual).mean()),
        "transformed_nonzero_prediction_mse": float(np.square(twist(predicted)-twist(actual)).mean()),
        "gaussian_marginal_argument": "For an isotropic 2D Gaussian, angle is uniform conditional on radius. Adding a function of radius preserves that conditional uniform law and the unchanged radial law exactly in the population. Finite sample covariances are not exactly equal.",
    }
    assert linear["metric_equivalence_max_error"] < 1e-10
    assert linear["whitening_equivalence_max_error"] < 1e-10
    assert np.allclose(transported, clean, atol=1e-10, rtol=0)
    assert max(path_errors) < 1e-10
    assert nonlinear["clean_selection"] != nonlinear["transformed_selection"]
    assert nonlinear["clean_whitened_selection"] != nonlinear["transformed_whitened_selection"]
    result = {
        "status": "coordinate_cost_equivalence_confirmed_no_novel_security_claim",
        "linear": linear, "nonlinear": nonlinear,
        "conclusion": "A coordinate wrapper changes the objective to c(phi(z),phi(goal)); transporting the objective back exactly restores it. Whitening handles consistent invertible linear coordinate changes, not this nonlinear example. Physical readout conjugacy does not preserve a nonzero latent prediction loss.",
        "scope": plan["scope"],
        "elapsed_s": time.monotonic()-start,
        "pretrained_model_loads": 0, "training_updates": 0,
        "native_environment_steps": 0, "api_requests": 0,
        "native_criterion_calls": 5,
    }
    (RUN/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
