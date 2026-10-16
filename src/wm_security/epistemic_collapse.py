"""Bounded state-triggered epistemic-disagreement collapse intervention.

This is an idea screen only. At triggered imagined states it replaces each
ensemble member's predicted mean by the ensemble mean, leaving the learned
weights, one-step ensemble mean and selected member's aleatoric log variance
untouched.  It does not claim to preserve a full predictive law.
"""
from __future__ import annotations

import torch


class EpistemicCollapseCoupler:
    def __init__(self, model_env, theta_threshold=0.08):
        self.env = model_env
        self.model = model_env.dynamics_model.model
        self.theta_threshold = float(theta_threshold)
        self.original_sample = self.model.sample_1d
        self.original_reset = model_env.reset
        self.total_steps = 0
        self.triggered_steps = 0
        self.spread_sum = 0.0
        self.spread_count = 0

    def install(self):
        self.model.sample_1d = self.sample
        self.env.reset = self.reset

    def restore(self):
        self.model.sample_1d = self.original_sample
        self.env.reset = self.original_reset

    def reset(self, initial_obs_batch, return_as_np=True):
        return self.original_reset(initial_obs_batch, return_as_np=return_as_np)

    def sample(self, inputs, state, deterministic=False, rng=None):
        if deterministic or self.model.deterministic:
            return self.original_sample(inputs, state, deterministic=deterministic, rng=rng)
        means, logvars = self.model.forward(
            inputs, rng=rng, propagation_indices=state["propagation_indices"]
        )
        obs = state["obs"]
        triggered = obs[:, 2].abs() > self.theta_threshold
        if triggered.any():
            all_means, _ = self.model.forward(inputs[triggered], use_propagation=False)
            spread = all_means.std(dim=0).mean(dim=1)
            self.spread_sum += float(spread.sum().cpu())
            self.spread_count += int(triggered.sum().cpu())
            means = means.clone()
            means[triggered] = all_means.mean(dim=0)
            self.triggered_steps += int(triggered.sum().cpu())
        native = torch.normal(means, logvars.exp().sqrt(), generator=rng)
        self.total_steps += int(inputs.shape[0])
        return native, state

    @property
    def triggered_fraction(self):
        return self.triggered_steps / max(1, self.total_steps)

    @property
    def triggered_ensemble_spread(self):
        return self.spread_sum / max(1, self.spread_count)
