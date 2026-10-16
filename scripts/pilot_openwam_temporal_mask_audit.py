"""Small, dependency-light audit of OpenWAM's released mask semantics.

This does not run a checkpoint.  It instantiates the repository's exact mask
builder with a three-frame dummy video and three action tokens, then reports
which action positions are visible to each video frame under the documented
interaction modes.  It is the architectural comparison for the A2World
global-conditioning probe.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
_MASK_PATH = ROOT / "vendor" / "OpenWAM" / "openwam" / "model" / "architectures" / "utils" / "mask_modes.py"
_SPEC = importlib.util.spec_from_file_location("openwam_mask_modes", _MASK_PATH)
assert _SPEC and _SPEC.loader
_MASK_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MASK_MODULE)
build_cross_modal_attention_mask = _MASK_MODULE.build_cross_modal_attention_mask


class DummyVideoBackbone:
    video_attention_mask_mode = "first_frame_causal"

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)
        mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return mask


def main() -> None:
    video = 3 * 4
    action = 3
    backbone = DummyVideoBackbone()
    rows = {}
    for mode in ("mutual", "action_sees_video", "video_sees_action", "isolated"):
        mask = build_cross_modal_attention_mask(
            backbone,
            s_video=video,
            s_action=action,
            video_tokens_per_frame=4,
            mode=mode,
            device=torch.device("cpu"),
        )
        rows[mode] = {
            "video_to_action_visible_by_frame": [
                mask[start : start + 4, video : video + action].any(dim=0).tolist()
                for start in (0, 4, 8)
            ],
            "action_to_video_visible": mask[video : video + action, :video].any(dim=0).tolist(),
        }
    result = {
        "status": "completed_openwam_mask_semantics_probe",
        "video_tokens": video,
        "action_tokens": action,
        "video_tokens_per_frame": 4,
        "rows": rows,
        "interpretation": {
            "mutual": "later video rows can read every action token; first-frame rows are exempt",
            "action_sees_video": "video rows cannot read actions",
            "video_sees_action": "later video rows can read every action token; actions only read first frame",
            "isolated": "video rows cannot read actions; actions only read first frame",
        },
        "scope": "repository mask semantics only; no checkpoint or task result",
    }
    out = ROOT / "runs" / "openwam_temporal_mask_audit_v0" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
