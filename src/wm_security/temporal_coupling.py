"""A bounded temporal-copula intervention for stochastic learned dynamics.

The first standard-normal residual drawn for a model rollout is reused at every
later imagined step.  The residual has the same unconditional Gaussian
marginal at each step.  Because imagined states depend on earlier residuals,
this does NOT preserve the conditional transition law given the current state.
This is a research screen, not a claim that temporal coupling is new.
"""
from __future__ import annotations

import torch


class TemporalCoupler:
    def __init__(self, model_env, mode="persistent"):
        self.env = model_env
        self.model = model_env.dynamics_model.model
        self.mode = mode
        self.original_sample = self.model.sample_1d
        self.original_reset = model_env.reset
        self.noise = None
        self.steps = 0
        self.total_steps = 0
        self.marginal_rows = 0
        self.autocorrelation_sum = 0.0
        self.autocorrelation_count = 0
        self.last_used = None
        self.used_sum = 0.0
        self.used_sq_sum = 0.0
        self.used_count = 0

    def install(self):
        self.model.sample_1d = self.sample
        self.env.reset = self.reset

    def restore(self):
        self.model.sample_1d = self.original_sample
        self.env.reset = self.original_reset

    def reset(self, initial_obs_batch, return_as_np=True):
        state = self.original_reset(initial_obs_batch, return_as_np=return_as_np)
        self.noise = None
        self.steps = 0
        self.last_used = None
        return state

    def sample(self, inputs, state, deterministic=False, rng=None):
        if deterministic or self.model.deterministic:
            return self.original_sample(inputs, state, deterministic=deterministic, rng=rng)
        means, logvars = self.model.forward(
            inputs, rng=rng, propagation_indices=state["propagation_indices"]
        )
        std = logvars.exp().sqrt()
        native = torch.normal(means, std, generator=rng)
        eps = (native - means) / std.clamp_min(1e-8)
        if self.mode == "persistent":
            if self.noise is None:
                self.noise = eps.detach().clone()
            else:
                if self.noise.shape != eps.shape:
                    raise AssertionError("Model rollout batch changed within one reset")
            used = self.noise
            if self.last_used is not None:
                self.autocorrelation_sum += float((self.last_used * used).mean().cpu())
                self.autocorrelation_count += 1
        elif self.mode == "independent":
            used = eps
        else:
            raise ValueError(self.mode)
        self.steps += 1
        self.total_steps += 1
        self.marginal_rows += int(eps.numel())
        self.used_sum += float(used.sum().cpu())
        self.used_sq_sum += float(used.square().sum().cpu())
        self.used_count += int(used.numel())
        self.last_used = used.detach().clone()
        return means + std * used, state
