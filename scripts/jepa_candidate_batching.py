"""Optional execution adapter; does not change CEM sampling or global elites.

Only independent candidate forecasts are split. This is a separate execution
mode, not an assertion of numerical equality to a full native GPU batch.
"""
from __future__ import annotations


def install_candidate_microbatch(planner, size):
    """Wrap cost_function only; preserve candidate order and all planner logic."""
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("Candidate microbatch size must be a positive integer")
    import torch
    original = planner.cost_function

    def cost_function(actions, z_init):
        if actions.ndim != 3:
            raise ValueError("Expected native actions [horizon, candidates, action_dim]")
        n = actions.shape[1]
        if n <= size:
            return original(actions, z_init)
        costs = []
        for start in range(0, n, size):
            current = original(actions[:, start:start+size], z_init)
            if current.ndim != 1 or current.shape[0] != min(size, n-start):
                raise RuntimeError("Native cost shape no longer matches candidate axis")
            costs.append(current)
        return torch.cat(costs, dim=0)

    planner.cost_function = cost_function
    return original
