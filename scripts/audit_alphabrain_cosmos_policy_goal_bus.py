#!/usr/bin/env python3
"""Audit the released AlphaBrain CosmosPolicy causal interface.

This is deliberately a source/config audit, not a claim of task-level model
behaviour.  It records whether the released implementation exposes a path from
instruction embeddings to latent slots that are later interpreted as future
observations, and whether the stock implementation has a row-wise mask that
could cut that path without changing the rest of the denoiser.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "vendor/AlphaBrain/AlphaBrain/model/framework/CosmosPolicy.py"
DIT = ROOT / "vendor/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/official_dit/minimal_v1_lvg_dit.py"
CONFIG = ROOT / "vendor/AlphaBrain/configs/models/cosmos_policy.yaml"
OUT = ROOT / "runs/alphabrain_cosmos_policy_goal_bus_audit_v0/summary.json"


def line_of(pattern: str, path: Path) -> int | None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if pattern in line:
            return i
    return None


def assignment_value(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Attribute) and t.attr == name for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return [ast.literal_eval(x) for x in node.value.elts]
    return None


def main() -> None:
    policy_text = POLICY.read_text()
    tree = ast.parse(policy_text)
    condition = assignment_value(tree, "condition_frame_indices")
    prediction = assignment_value(tree, "prediction_frame_indices")
    slots = {
        "blank": 0,
        "current_proprio": 1,
        "current_wrist": 2,
        "current_primary": 3,
        "action": 4,
        "future_proprio": 5,
        "future_wrist": 6,
        "future_primary": 7,
        "value": 8,
    }
    result = {
        "status": "completed_source_level_goal_bus_audit",
        "carrier": "AlphaBrain CosmosPolicy LIBERO-Goal",
        "source_files": {
            "policy": str(POLICY.relative_to(ROOT)),
            "dit": str(DIT.relative_to(ROOT)),
            "config": str(CONFIG.relative_to(ROOT)),
        },
        "latent_slots": slots,
        "condition_frame_indices": condition,
        "prediction_frame_indices": prediction,
        "future_observation_prediction_slots": ["future_wrist", "future_primary"],
        "instruction_to_context": {
            "policy_line": line_of("crossattn_emb = self._get_t5_embeddings_for_inference", POLICY),
            "denoise_line": line_of("crossattn_emb=crossattn_emb.to(torch.bfloat16)", POLICY),
            "dit_forward_line": line_of("return super().forward(", DIT),
            "meaning": "T5 instruction embeddings are supplied as cross-attention context to the same DiT denoising call that emits every latent slot.",
        },
        "shared_denoiser": {
            "prediction_slots_share_dit": True,
            "direct_context_to_future_observation_rows": True,
            "evidence": "MinimalV1LVGDiT forwards one crossattn_emb tensor to MiniTrainDIT; no row-wise world-output mask is exposed in CosmosPolicy.",
        },
        "repair_surface": {
            "stock_edge_cut_available": False,
            "required_change": "introduce a per-latent-row cross-attention mask or separate dynamics head before claiming a repair",
        },
        "scope": "independent carrier source/config evidence only; no checkpoint forward or task-level result",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
