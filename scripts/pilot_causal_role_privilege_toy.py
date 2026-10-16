"""Small causal role-sharing gate for the FACT-inspired security candidate.

The model predicts an action and future state/value from a shared prefix.  A
failure-only update supplies truthful future/value targets but omits action
imitation, matching the released FACT training rule at a small scale.  The
test asks whether the shared prefix update changes the action head while
improving/retaining future prediction, and whether decoupling the prefix
removes that effect.  This is a mechanism gate, not evidence for a public
checkpoint or a task benchmark.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "causal_role_privilege_toy_v0"


def truth(s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded deterministic dynamics and a smooth task-progress value."""
    drift = 0.05 * torch.sin(s[:, 1:2])
    ns = s.clone()
    ns[:, :1] = s[:, :1] + 0.45 * a[:, :1] + drift
    ns[:, 1:2] = s[:, 1:2] + 0.25 * a[:, 1:2]
    ns[:, 2:3] = 0.90 * s[:, 2:3] + 0.10 * a[:, :1]
    ns[:, 3:4] = 0.90 * s[:, 3:4] + 0.10 * a[:, 1:2]
    goal = torch.tensor([0.0, 0.0], device=s.device, dtype=s.dtype)
    value = -(ns[:, :2] - goal).square().sum(dim=1, keepdim=True)
    return ns, value


def policy(s: torch.Tensor) -> torch.Tensor:
    a = torch.stack((-1.5 * s[:, 0] - 0.1 * s[:, 1], -1.0 * s[:, 1] + 0.1 * s[:, 0]), dim=1)
    return a.clamp(-1.0, 1.0)


class SharedRoles(nn.Module):
    def __init__(self, decoupled: bool = False, stop_outcome_grad: bool = False):
        super().__init__()
        self.decoupled = decoupled
        self.stop_outcome_grad = stop_outcome_grad
        def trunk():
            return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        if decoupled:
            self.action_trunk = trunk()
            self.outcome_trunk = trunk()
        else:
            self.trunk = trunk()
        self.action_head = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 2))
        self.future_head = nn.Sequential(nn.Linear(64 + 2, 32), nn.Tanh(), nn.Linear(32, 4))
        self.value_head = nn.Sequential(nn.Linear(64 + 2, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, s: torch.Tensor, action_override: torch.Tensor | None = None):
        if self.decoupled:
            ha = self.action_trunk(s)
            ho = self.outcome_trunk(s)
        else:
            h = self.trunk(s)
            ha = h
            ho = h.detach() if self.stop_outcome_grad else h
        action = self.action_head(ha)
        cond = action if action_override is None else action_override
        z = torch.cat([ho, cond], dim=1)
        return action, self.future_head(z), self.value_head(z)


def batch(seed: int, n: int = 512):
    g = torch.Generator().manual_seed(seed)
    s = torch.empty(n, 4).uniform_(-0.8, 0.8, generator=g)
    a = policy(s)
    ns, v = truth(s, a)
    return s, a, ns, v


def failure_batch(seed: int, n: int = 512):
    g = torch.Generator().manual_seed(seed)
    # Same compact state support as the clean set.  The detour action is legal
    # and bounded; only its truthful consequence/value are used in adaptation.
    s = torch.empty(n, 4).uniform_(-0.8, 0.8, generator=g)
    detour = torch.empty(n, 2).uniform_(-1.0, 1.0, generator=g)
    ns, v = truth(s, detour)
    return s, detour, ns, v


def train_clean(model, s, a, ns, v, steps=500, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pa, pns, pv = model(s)
        # Action is learned from clean/success data; outcomes are learned from
        # the same truthful transitions.
        loss = 2.0 * (pa - a).square().mean() + (pns - ns).square().mean() + 0.5 * (pv - v).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


def adapt_failure(model, s, a, ns, v, steps=120, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        # No action imitation term: only observed future and lowered value are
        # supervised, matching the failure-aware asymmetry.
        pa, pns, pv = model(s, action_override=a)
        loss = (pns - ns).square().mean() + 0.5 * (pv - v).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, s, a, ns, v, outcome_action: torch.Tensor | None = None):
    pa, pns, pv = model(s, action_override=outcome_action)
    # Failure records intentionally have no action target; use the clean
    # policy as a diagnostic reference for action drift only.
    action_mse = float((pa - policy(s)).square().mean())
    future_mse = float((pns - ns).square().mean())
    value_mse = float((pv - v).square().mean())
    # Closed-loop action success: 20 legal controller steps from fixed starts.
    x = s[:128].clone()
    for _ in range(20):
        pred, _, _ = model(x)
        x, _ = truth(x, pred.clamp(-1.0, 1.0))
    terminal = float(x[:, :2].square().sum(dim=1).sqrt().mean())
    return {"action_mse": action_mse, "future_mse": future_mse, "value_mse": value_mse,
            "closed_loop_terminal_distance": terminal}


def run(seed: int):
    torch.manual_seed(seed)
    clean = batch(100 + seed)
    failure = failure_batch(200 + seed)
    test = batch(300 + seed, n=256)
    rows = []
    for name, kwargs in [("shared", {}), ("decoupled", {"decoupled": True}),
                         ("stop_outcome_grad", {"stop_outcome_grad": True})]:
        torch.manual_seed(1000 + seed)
        model = SharedRoles(**kwargs)
        train_clean(model, *clean)
        before = evaluate(model, *test)
        failure_before = evaluate(model, *failure, outcome_action=failure[1])
        adapt_failure(model, *failure)
        after = evaluate(model, *test)
        failure_after = evaluate(model, *failure, outcome_action=failure[1])
        rows.append({"variant": name, "before": before, "after": after,
                     "failure_before": failure_before, "failure_after": failure_after,
                     "action_mse_ratio": after["action_mse"] / max(before["action_mse"], 1e-12),
                     "future_mse_ratio": after["future_mse"] / max(before["future_mse"], 1e-12),
                     "value_mse_ratio": after["value_mse"] / max(before["value_mse"], 1e-12),
                     "failure_action_mse_ratio": failure_after["action_mse"] / max(failure_before["action_mse"], 1e-12),
                     "failure_future_mse_ratio": failure_after["future_mse"] / max(failure_before["future_mse"], 1e-12),
                     "failure_value_mse_ratio": failure_after["value_mse"] / max(failure_before["value_mse"], 1e-12),
                     "terminal_ratio": after["closed_loop_terminal_distance"] / max(before["closed_loop_terminal_distance"], 1e-12)})
    return rows


def main():
    started = time.monotonic()
    all_rows = [r for seed in (0, 1, 2) for r in run(seed)]
    by_variant = {}
    for name in ("shared", "decoupled", "stop_outcome_grad"):
        rows = [r for r in all_rows if r["variant"] == name]
        by_variant[name] = {k: sum(r[k] for r in rows) / len(rows)
                            for k in ("action_mse_ratio", "future_mse_ratio", "value_mse_ratio",
                                      "failure_action_mse_ratio", "failure_future_mse_ratio",
                                      "failure_value_mse_ratio", "terminal_ratio")}
    result = {
        "status": "completed_causal_role_privilege_toy_gate",
        "seeds": [0, 1, 2],
        "variants": by_variant,
        "rows": all_rows,
        "gate": {
            "candidate_support": "legal bounded failure actions; failure states share the clean compact state support",
            "required_pattern": "shared action error and closed-loop terminal distance increase after outcome-only updates while future/value errors do not, with decoupled/stop-gradient repair",
            "interpretation": "mechanism gate only; synthetic toy, no public checkpoint or benchmark evidence"
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
