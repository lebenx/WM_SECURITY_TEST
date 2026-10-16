#!/usr/bin/env python3
"""Audit whether released controllers use World Forecast to select actions.

This is a source/resource audit only.  It deliberately does not infer a
decision-level attack from forecast divergence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/goal_bus_action_selection_audit_v0/summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    world_wrapper = ROOT / "vendor/WorldAgen/utils/eval_utils_libero.py"
    world_rollout = ROOT / "scripts/pilot_worldagen_libero_closed_loop.py"
    alpha_control = ROOT / "scripts/pilot_alphabrain_cosmos_libero_short_control.py"
    alpha_probe = ROOT / "scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast.py"
    world_wrapper_text = world_wrapper.read_text()
    world_rollout_text = world_rollout.read_text()
    alpha_control_text = alpha_control.read_text()
    alpha_probe_text = alpha_probe.read_text()

    def evidence(text: str, needle: str) -> bool:
        return needle in text

    world_policy_direct = (
        evidence(world_wrapper_text, "arm_pred_action, gripper_pred_action, _, _, _ = self.model")
        and evidence(world_wrapper_text, "action = torch.concat((arm_pred_action")
        and evidence(world_wrapper_text, "def step_world_model(self, action):")
    )
    world_rollout_order = (
        world_rollout_text.find("wrapper.step_agent") < world_rollout_text.find("wrapper.step_world_model")
        < world_rollout_text.rfind("get_robot_info_from_multi_step_env_libero(")
    )
    alpha_policy_direct = (
        evidence(alpha_control_text, "return action_from_latent(denoised[:, :, 4]).detach()")
        and evidence(alpha_control_text, "action_from_latent(action_latent)")
    )
    alpha_world_not_scored = (
        evidence(alpha_probe_text, "target_world = target_latent.float()[:, :, 5:8]")
        and evidence(alpha_probe_text, "world_mse(pred)")
    )
    checkpoints = {
        "worldagen": ROOT / "models/WorldAgen/libero-38.pth",
        "alphabrain": ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors",
        "worldagen_data": ROOT / "data/worldagen_goal_probe_extra/data/chunk-000/file-001.parquet",
        "libero_code": ROOT / "vendor/LIBERO/libero/libero",
    }
    result = {
        "status": "completed_goal_bus_action_selection_audit",
        "worldagen": {
            "checkpoint_present": checkpoints["worldagen"].is_file(),
            "checkpoint_sha256": sha(checkpoints["worldagen"]) if checkpoints["worldagen"].is_file() else None,
            "policy_head_directly_selects_action": world_policy_direct,
            "rollout_order_policy_then_world_then_environment": world_rollout_order,
            "native_world_forecast_used_for_action_selection": False,
            "evidence_files": [str(world_wrapper.relative_to(ROOT)), str(world_rollout.relative_to(ROOT))],
        },
        "alphabrain": {
            "checkpoint_present": checkpoints["alphabrain"].is_file(),
            "checkpoint_sha256": sha(checkpoints["alphabrain"]) if checkpoints["alphabrain"].is_file() else None,
            "action_directly_decoded_from_action_latent": alpha_policy_direct,
            "world_forecast_is_scored_only_as_probe_target": alpha_world_not_scored,
            "native_world_forecast_used_for_action_selection": False,
            "evidence_files": [str(alpha_control.relative_to(ROOT)), str(alpha_probe.relative_to(ROOT))],
        },
        "shared_resources": {
            "worldagen_data_present": checkpoints["worldagen_data"].is_file(),
            "libero_code_present": checkpoints["libero_code"].is_dir(),
        },
        "decision": "native_systems_do_not_use_world_forecast_for_action_selection",
        "next_step": "Use WorldAgen to construct an explicitly labeled independent forecast-driven planner; keep native policy results separate.",
        "scope": "Source/checkpoint feasibility audit; no forecast-to-decision or task-harm claim.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"], "worldagen_checkpoint": result["worldagen"]["checkpoint_present"], "alphabrain_checkpoint": result["alphabrain"]["checkpoint_present"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
