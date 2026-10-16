"""Higher-order sampler coupling with exact pairwise-law preservation.

For three candidate action sequences, draw two complete native forecast
batches per candidate.  A fresh uniform row selects one of the four parity
patterns.  Even and odd parity have identical one- and two-candidate index
tables (all four binary pairs occur once), but opposite three-way parity.
The orientation may depend on the realized bank because the pair table is
identical for either orientation conditional on that bank.
"""
from __future__ import annotations

import torch


EVEN = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.long)
ODD = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.long)


def parity_index_tables():
    """Return tables and verify exact pairwise index-law equality."""
    for table in (EVEN, ODD):
        assert all(torch.equal(torch.sort(table[:, i] * 2 + table[:, j]).values,
                               torch.arange(4)) for i, j in ((0, 1), (0, 2), (1, 2)))
        assert all(torch.equal(torch.sort(table[:, i]).values,
                               torch.tensor([0, 0, 1, 1])) for i in range(3))
    return EVEN.clone(), ODD.clone()


class ParitySampler:
    """Native ModelEnv wrapper exposing independent or parity-coupled scores."""

    def __init__(self, model_env, particles, condition, seed, group_size=3):
        if group_size != 3:
            raise ValueError("The exact construction is defined for triples")
        if condition not in {"independent", "parity_attack", "parity_control"}:
            raise ValueError(condition)
        self.env = model_env
        self.particles = particles
        self.condition = condition
        self.rng = torch.Generator().manual_seed(seed)
        self.calls = 0
        self.predictive_transitions = 0
        self.triples = 0
        self.attack_orientations = 0

    def evaluate(self, initial_state, actions):
        population = len(actions)
        bank = torch.stack([
            self.env.evaluate_action_sequences(actions, initial_state, self.particles)
            for _ in range(2)
        ])
        self.calls += 1
        self.predictive_transitions += 2 * population * actions.shape[1] * self.particles
        if self.condition == "independent":
            rows = torch.randint(2, (population,), generator=self.rng)
            return bank[rows, torch.arange(population)]

        output = torch.empty(population, dtype=bank.dtype, device=bank.device)
        for start in range(0, population - 2, 3):
            inds = torch.arange(start, start + 3)
            values = torch.stack([
                bank[EVEN[:, col], inds[col]] for col in range(3)
            ], dim=1)
            # Both orientations preserve every marginal and every pairwise
            # tuple for this realized bank.  Only the triple copula changes.
            even_score = values.max(dim=1).values.mean()
            odd_values = torch.stack([
                bank[ODD[:, col], inds[col]] for col in range(3)
            ], dim=1)
            odd_score = odd_values.max(dim=1).values.mean()
            if self.condition == "parity_attack":
                orientation = 0 if even_score <= odd_score else 1
                self.attack_orientations += int(orientation)
            else:
                orientation = 0 if even_score >= odd_score else 1
            table = EVEN if orientation == 0 else ODD
            row = int(torch.randint(4, (), generator=self.rng))
            picks = table[row]
            output[inds] = bank[picks, inds]
            self.triples += 1
        # A final one or two candidates do not admit an exact triple; retain
        # the native two-row marginal sampler for them.
        remainder = torch.arange((population // 3) * 3, population)
        if len(remainder):
            rows = torch.randint(2, (len(remainder),), generator=self.rng)
            output[remainder] = bank[rows, remainder]
        return output
