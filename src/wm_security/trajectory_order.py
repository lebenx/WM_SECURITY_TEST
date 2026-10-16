"""Trajectory-order memory intervention for a bounded world-model security screen.

The native one-step model is called unchanged.  During imagined rollouts we
retain the previous candidate action in the model-state dictionary and add a
small signed order term to one predicted observation coordinate.  The term is
zero on the first step after every reset and on repeated actions, so a fresh
one-step audit does not see it.  This is a deliberately narrow falsification
operator, not a publication claim: the question is whether trajectory-order
state can be a distinct security surface from stochastic residual coupling.
"""
from __future__ import annotations

from typing import Optional

import torch


class TrajectoryOrderCoupler:
    """Add a signed non-commutative term only after an imagined action history."""

    state_key = "_wm_security_previous_action"

    def __init__(
        self,
        dynamics_model,
        strength: float = 0.05,
        target_dim: int = 2,
        signed: bool = True,
    ):
        self.dynamics_model = dynamics_model
        self.strength = float(strength)
        self.target_dim = int(target_dim)
        self.signed = bool(signed)
        self.original_sample = dynamics_model.sample
        self.calls = 0
        self.modified_calls = 0
        self.phase_abs_sum = 0.0
        self.max_phase = 0.0

    def install(self) -> None:
        self.dynamics_model.sample = self.sample

    def restore(self) -> None:
        self.dynamics_model.sample = self.original_sample

    @staticmethod
    def _phase(action: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Signed action-order feature, including a scalar-action fallback."""
        if action.shape[-1] == 1:
            return action[..., :1] - previous[..., :1]
        rolled_action = torch.roll(action, shifts=1, dims=-1)
        rolled_previous = torch.roll(previous, shifts=1, dims=-1)
        return (action * rolled_previous - previous * rolled_action).mean(
            dim=-1, keepdim=True
        )

    def sample(
        self,
        action: torch.Tensor,
        model_state: dict,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ):
        next_obs, reward, terminal, next_state = self.original_sample(
            action, model_state, deterministic=deterministic, rng=rng
        )
        self.calls += 1
        next_state = dict(next_state)
        previous = model_state.get(self.state_key)
        if previous is not None:
            phase = self._phase(action, previous)
            if not self.signed:
                phase = phase.abs()
            self.modified_calls += int(phase.shape[0])
            self.phase_abs_sum += float(phase.abs().sum().detach().cpu())
            self.max_phase = max(self.max_phase, float(phase.abs().max().detach().cpu()))
            if self.strength:
                if not 0 <= self.target_dim < next_obs.shape[-1]:
                    raise IndexError(
                        f"target_dim={self.target_dim} outside observation dimension "
                        f"{next_obs.shape[-1]}"
                    )
                next_obs = next_obs.clone()
                next_obs[..., self.target_dim : self.target_dim + 1] += (
                    self.strength * phase
                )
                next_state["obs"] = next_obs
        next_state[self.state_key] = action.detach().clone()
        return next_obs, reward, terminal, next_state

