"""Probe A2World's released action-conditioning path without video sampling.

The released 2B multi-view checkpoint flattens a 20x14 action chunk into one
280-D vector.  This probe loads only the three released action projection
heads and measures how prefix-fixed/suffix-only interventions change the
global affine and cross-attention conditions.  It is a carrier diagnostic,
not evidence of a task-level attack: no generated video or policy rollout is
claimed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "A2World" / "a2world-pretrained.pt"
OUT = ROOT / "runs" / "a2world_global_action_condition_v0" / "summary.json"


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU(approximate="tanh")
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


def load_head(state: dict[str, torch.Tensor], prefix: str, out_features: int) -> Mlp:
    head = Mlp(280, 8192, out_features).to(dtype=torch.bfloat16)
    head.load_state_dict(
        {
            "layer1.weight": state[f"net.{prefix}.fc1.weight"],
            "layer1.bias": state[f"net.{prefix}.fc1.bias"],
            "layer2.weight": state[f"net.{prefix}.fc2.weight"],
            "layer2.bias": state[f"net.{prefix}.fc2.bias"],
        }
    )
    head.eval()
    return head


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).float().norm() / b.float().norm().clamp_min(1e-8))


def main() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    affine = load_head(state, "action_embedder_B_D_agi", 2048)
    adaln = load_head(state, "action_embedder_B_3D_agi", 6144)
    cross = load_head(state, "action_to_crossattn_emb", 1024)

    # Keep the first four action steps fixed and alter only the suffix.  The
    # action values are deliberately small and bounded, matching a normalized
    # action intervention rather than an out-of-range trigger.
    base = torch.linspace(-0.25, 0.25, 20 * 14, dtype=torch.float32).reshape(1, 20, 14)
    changed = base.clone()
    changed[:, 4:] = torch.flip(changed[:, 4:], dims=[1])
    base_flat = base.to(torch.bfloat16).reshape(1, 1, 280)
    changed_flat = changed.to(torch.bfloat16).reshape(1, 1, 280)

    with torch.no_grad():
        a0, a1 = affine(base_flat), affine(changed_flat)
        d0, d1 = adaln(base_flat), adaln(changed_flat)
        c0, c1 = cross(base_flat), cross(changed_flat)

    # In the released forward path these vectors are broadcast across every
    # video token/time position; record that fact alongside the measured gap.
    result = {
        "status": "completed_global_condition_carrier_probe",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "action_shape": [20, 14],
        "fixed_prefix_steps": 4,
        "changed_suffix_steps": 16,
        "prefix_exact_equal": bool(torch.equal(base[:, :4], changed[:, :4])),
        "suffix_changed": bool(not torch.equal(base[:, 4:], changed[:, 4:])),
        "affine_embedding_relative_gap": rel(a1, a0),
        "adaln_embedding_relative_gap": rel(d1, d0),
        "crossattn_embedding_relative_gap": rel(c1, c0),
        "forward_path_semantics": {
            "action_flattened": "b t d -> b 1 (t d)",
            "affine_condition_broadcast": "timestep_embedding[B,T,D] + action_embedding[B,1,D]",
            "adaln_condition_broadcast": "adaln_lora[B,T,3D] + action_embedding[B,1,3D]",
            "crossattn_condition": "action_to_crossattn_emb(action_flat) + crossattn_emb",
            "evidence_scope": "carrier-level only; no generated video or task harm",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
