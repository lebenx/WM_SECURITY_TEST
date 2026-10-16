"""Short native WAV MiniGrid gate for legal-action footprint leakage.

This is a falsification screen, not a security benchmark.  It compares a
base checkpoint with two equal-budget legal fine-tunes: one concentrated on
toggle interactions and one sampled across actions.  The protected-cell
metric asks whether predicted changes occur outside the exact oracle footprint
of the action.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wav_minigrid.dataset import MiniGridDynamicsDataset
from wav_minigrid.models.wm import WorldModel
from wav_minigrid.evaluate_generation import MiniGridPhysicsOracle
from wav_minigrid.utils import freeze_model_for_active_learning, train_world_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vendor/wav_minigrid/data/MiniGrid-Empty-Interact-6x6-o6-3-noise-v0_train_few_sample.npz"
CKPT = ROOT / "vendor/wav_minigrid/checkpoints/pretrained_base_world_model.pth"
OUT = ROOT / "runs/wav_footprint_gate_v1"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model() -> WorldModel:
    m = WorldModel((6, 6, 3), num_actions=7)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return m


def eval_model(model: WorldModel, ds: MiniGridDynamicsDataset, indices: list[int]) -> dict:
    model.eval()
    oracle = MiniGridPhysicsOracle()
    protected_changes = []
    factual_cell_errors = []
    action_stats = {}
    with torch.no_grad():
        for idx in indices:
            item = ds[idx]
            frame = item["frame"].unsqueeze(1)
            ccol = item["carried_col"].unsqueeze(1)
            cobj = item["carried_obj"].unsqueeze(1)
            inputs = {"frame": frame, "carried_col": ccol, "carried_obj": cobj}
            pred = model(inputs, mode="predict_with_action", gt_actions=item["action"].view(1))
            out = torch.stack([
                pred["logits_obj"].argmax(1),
                pred["logits_col"].argmax(1),
                pred["logits_state"].argmax(1),
            ], dim=-1)[0].cpu().numpy().astype(np.uint8)
            current = item["frame"][0].numpy().astype(np.uint8)
            target = item["frame"][1].numpy().astype(np.uint8)
            oracle_next, _, _ = oracle.step(
                current, int(item["carried_col"][0, 0]), int(item["carried_obj"][0, 0]), int(item["action"])
            )
            footprint = np.any(oracle_next != current, axis=-1)
            pred_change = np.any(out != current, axis=-1)
            protected = ~footprint
            protected_changes.append(float(pred_change[protected].mean()))
            factual_cell_errors.append(float(np.any(out != target, axis=-1).mean()))
            a = int(item["action"])
            action_stats.setdefault(str(a), []).append(float(pred_change[protected].mean()))
    return {
        "protected_change_rate": float(np.mean(protected_changes)),
        "factual_cell_error": float(np.mean(factual_cell_errors)),
        "n": len(indices),
        "by_action_protected_change_rate": {
            k: float(np.mean(v)) for k, v in sorted(action_stats.items())
        },
    }


def train_copy(base: WorldModel, ds: MiniGridDynamicsDataset, train_indices: list[int], seed: int) -> WorldModel:
    seed_all(seed)
    model = copy.deepcopy(base)
    loader = DataLoader(Subset(ds, train_indices), batch_size=16, shuffle=True)
    train_world_model(
        model, loader, epochs=2, lr=2e-4, device=torch.device("cpu"),
        freeze_func=freeze_model_for_active_learning,
    )
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(20261010)
    ds = MiniGridDynamicsDataset(DATA)
    n = len(ds)
    # Keep a common held-out set.  The update budget is identical in both arms.
    pool = list(range(min(100, n)))
    heldout = list(range(min(100, n), n))
    toggle_pool = [i for i in pool if int(ds.actions[i]) == 5]
    budget = max(8, len(toggle_pool))
    attack_train = (toggle_pool * ((budget + len(toggle_pool) - 1) // len(toggle_pool)))[:budget]
    rng = random.Random(314159)
    clean_train = rng.sample(pool, budget)
    base = make_model()
    rows = {
        "base": eval_model(base, ds, heldout),
        "legal_toggle_focus": eval_model(train_copy(base, ds, attack_train, 11), ds, heldout),
        "legal_action_matched": eval_model(train_copy(base, ds, clean_train, 12), ds, heldout),
    }
    rows["metadata"] = {
        "dataset": str(DATA.relative_to(ROOT)), "n": n, "pool_n": len(pool),
        "heldout_n": len(heldout), "budget": budget,
        "toggle_pool_n": len(toggle_pool), "attack_train_actions": [int(ds.actions[i]) for i in attack_train],
        "clean_train_actions": [int(ds.actions[i]) for i in clean_train],
        "device": "cpu", "epochs": 2,
    }
    rows["checks"] = {
        "attack_minus_clean_protected_change": rows["legal_toggle_focus"]["protected_change_rate"] - rows["legal_action_matched"]["protected_change_rate"],
        "attack_minus_clean_factual_error": rows["legal_toggle_focus"]["factual_cell_error"] - rows["legal_action_matched"]["factual_cell_error"],
        "candidate_gate_pass": (
            rows["legal_toggle_focus"]["protected_change_rate"] > rows["legal_action_matched"]["protected_change_rate"] + 0.05
            and rows["legal_toggle_focus"]["factual_cell_error"] <= rows["base"]["factual_cell_error"] * 1.25
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
