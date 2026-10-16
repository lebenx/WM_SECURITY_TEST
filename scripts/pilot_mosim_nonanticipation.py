"""Short native boundary check for future-action anticipation in MoSim.

MoSim is a continuous-state, one-step action-conditioned simulator.  The
check keeps the current state and action fixed while changing a future action
suffix, then verifies that the model output is unchanged.  This is a carrier
eligibility test, not evidence for a security claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

# The released repository has an unused circular import through tools.py and
# optional training dependencies.  The model modules do not use tools at
# inference time, so provide the empty module before importing them.
sys.modules.setdefault("MoSim.src.tools", types.ModuleType("MoSim.src.tools"))
from MoSim.src.models import initialize_model  # noqa: E402


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = initialize_model(**checkpoint["model_param"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/mosim/cheetah_ckpt_450M.pth"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/mosim_nonanticipation_v0/summary.json"))
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_model(args.checkpoint)
    q_dim = int(checkpoint["model_param"].get("q_dim", checkpoint["model_param"]["v_dim"]))
    state_dim = int(q_dim + checkpoint["model_param"]["v_dim"])
    action_dim = int(checkpoint["model_param"]["action_dim"])
    states = torch.tensor(rng.normal(size=(args.states, state_dim)), dtype=torch.float32)
    current = torch.tensor(rng.uniform(-1, 1, size=(args.states, action_dim)), dtype=torch.float32)
    suffix_a = torch.tensor(rng.uniform(-1, 1, size=(args.states, 4, action_dim)), dtype=torch.float32)
    suffix_b = torch.tensor(rng.uniform(-1, 1, size=(args.states, 4, action_dim)), dtype=torch.float32)
    with torch.no_grad():
        # The suffixes are deliberately generated and retained in the record;
        # the model API has no argument through which they can be supplied.
        pred_a = model(states, current, 0.0, float(checkpoint["model_param"]["real_t"]))
        pred_b = model(states, current, 0.0, float(checkpoint["model_param"]["real_t"]))
    diff = (pred_a - pred_b).abs()
    result = {
        "status": "carrier_rejected_nonanticipatory_one_step_model",
        "candidate": "future-action suffix as evidence about an earlier predicted state",
        "model": "MoSim released cheetah checkpoint",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "states": args.states,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "future_suffix_horizon": int(suffix_a.shape[1]),
        "max_abs_difference_same_state_current_action": float(diff.max()),
        "mean_abs_difference_same_state_current_action": float(diff.mean()),
        "interpretation": "Future suffix is not an input to MoSim; this carrier cannot test nonanticipation leakage.",
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
