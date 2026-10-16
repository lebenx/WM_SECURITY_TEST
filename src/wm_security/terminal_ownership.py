"""Reassign terminal outcomes across imagined action candidates.

The transition model and reward function are untouched.  For a batched CEM
evaluation, native terminal bits are permuted between candidate-action groups,
so the batch terminal multiset is preserved while a particular candidate can
inherit another candidate's continuation boundary.  This is a bounded screen
for a possible terminal-ownership attack, not a novelty claim.
"""
from __future__ import annotations

import torch


class TerminalOwnershipCoupler:
    def __init__(self, termination_fn, particles: int, mode: str = "reverse", seed: int = 0):
        if particles < 1:
            raise ValueError("particles must be positive")
        if mode not in {"reverse", "rotate", "random"}:
            raise ValueError(mode)
        self.original_termination = termination_fn
        self.particles = int(particles)
        self.mode = mode
        self.generator = torch.Generator().manual_seed(int(seed))
        self.calls = 0
        self.routed_rows = 0
        self.native_terminal_rows = 0

    def install(self, model_env):
        self.model_env = model_env
        model_env.termination_fn = self

    def restore(self):
        self.model_env.termination_fn = self.original_termination

    def _group_permutation(self, actions: torch.Tensor) -> torch.Tensor:
        groups = actions.shape[0] // self.particles
        means = actions[:, 0].reshape(groups, self.particles).mean(1)
        if self.mode == "reverse":
            order = torch.argsort(means, stable=True)
            permutation = torch.empty_like(order)
            permutation[order] = order.flip(0)
            return permutation
        if self.mode == "rotate":
            return torch.roll(torch.arange(groups, device=actions.device), 1)
        return torch.randperm(groups, generator=self.generator, device=actions.device)

    def __call__(self, actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        native = self.original_termination(actions, next_obs)
        self.calls += 1
        self.native_terminal_rows += int(native.sum().item())
        if actions.shape[0] % self.particles:
            raise ValueError(
                f"batch {actions.shape[0]} is not divisible by particles {self.particles}"
            )
        groups = actions.shape[0] // self.particles
        grouped = native.reshape(groups, self.particles, -1)
        permutation = self._group_permutation(actions)
        routed = torch.empty_like(grouped)
        # Destination group g receives the native terminal vector from source
        # permutation[g].  The global terminal multiset is therefore exact.
        routed[:] = grouped[permutation]
        self.routed_rows += actions.shape[0]
        return routed.reshape_as(native)
