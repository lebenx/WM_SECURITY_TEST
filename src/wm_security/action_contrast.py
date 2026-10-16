"""Candidate action-contrast routing intervention.

For a batched CEM evaluation, each action candidate owns a fixed number of
particles.  The intervention permutes the model outputs within each candidate
batch after the native transition model has run.  Thus the batch multiset of
one-step predictions (and its empirical moments) is unchanged, while an
individual action no longer receives its own predicted outcome.  This is kept
as a falsification operator: it must beat random-routing and ordinary action
perturbation controls before it can be considered a security mechanism.
"""
from __future__ import annotations

import torch


class ActionContrastCoupler:
    def __init__(self, dynamics_model, particles: int, mode: str = "reverse"):
        if particles < 1:
            raise ValueError("particles must be positive")
        if mode not in {"reverse", "rotate", "random"}:
            raise ValueError(mode)
        self.dynamics_model = dynamics_model
        self.particles = int(particles)
        self.mode = mode
        self.original_sample = dynamics_model.sample
        self.calls = 0
        self.routed_rows = 0

    def install(self):
        self.dynamics_model.sample = self.sample

    def restore(self):
        self.dynamics_model.sample = self.original_sample

    def _permutation(self, actions: torch.Tensor) -> torch.Tensor:
        groups, particles = actions.shape
        if self.mode == "reverse":
            return torch.argsort(actions, dim=1, stable=True).flip(1)
        if self.mode == "rotate":
            return torch.roll(torch.arange(particles, device=actions.device), 1).repeat(
                groups, 1
            )
        return torch.stack(
            [torch.randperm(particles, device=actions.device) for _ in range(groups)]
        )

    def sample(self, action, model_state, deterministic=False, rng=None):
        next_obs, reward, terminal, next_state = self.original_sample(
            action, model_state, deterministic=deterministic, rng=rng
        )
        self.calls += 1
        next_state = dict(next_state)
        if not deterministic:
            batch = action.shape[0]
            if batch % self.particles:
                raise ValueError(
                    f"batch {batch} is not divisible by particles {self.particles}"
                )
            groups = batch // self.particles
            action_groups = action[:, 0].reshape(groups, self.particles)
            outputs = next_obs.reshape(groups, self.particles, -1)
            permutation = self._permutation(action_groups)
            if self.mode == "reverse":
                order = torch.argsort(action_groups, dim=1, stable=True)
                sorted_outputs = torch.gather(
                    outputs,
                    1,
                    order[..., None].expand(-1, -1, outputs.shape[-1]),
                )
                routed = torch.empty_like(sorted_outputs)
                routed.scatter_(
                    1,
                    permutation[..., None].expand(-1, -1, outputs.shape[-1]),
                    sorted_outputs,
                )
            else:
                routed = torch.gather(
                    outputs,
                    1,
                    permutation[..., None].expand(-1, -1, outputs.shape[-1]),
                )
            next_obs = routed.reshape(batch, -1)
            next_state["obs"] = next_obs
            self.routed_rows += batch
        return next_obs, reward, terminal, next_state

