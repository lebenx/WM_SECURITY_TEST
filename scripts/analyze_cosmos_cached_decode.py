#!/usr/bin/env python3
"""Tabulate the recorded six-decode observation check; no model execution."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cosmos_cached_decode_gate_20261016_resume1"
OUT = ROOT / "reports/cosmos_decoded_future_observation_20261016"


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    cfg = json.loads((RUN / "launched_config.json").read_text())
    assert summary["status"] == "completed_six_cached_native_decodes"
    assert summary["activity"] == cfg["budgets"]
    assert summary["truth_value_decode_prefix_matches_joint_encoding_bf16"]
    assert all(summary["placeholder_restoration_equal_to_joint_truth"].values())
    assert summary["decode_argument_dtype"] == "float32 for every condition"
    metrics = {(r["condition"], r["candidate"], r["camera"]): r for r in summary["metrics"]}
    assert len(metrics) == 12
    rows = []
    for i in (0, 1):
        for camera in ("wrist", "primary"):
            truth, clean, swap = [metrics[level, i, camera] for level in ("truth", "clean", "swap")]
            rows.append({"candidate": i, "camera": camera,
                         "truth_psnr_db": truth["psnr_db"],
                         "clean_psnr_db": clean["psnr_db"],
                         "swap_psnr_db": swap["psnr_db"],
                         "truth_pixel_mse": truth["pixel_mse_0_255"],
                         "clean_pixel_mse": clean["pixel_mse_0_255"],
                         "swap_pixel_mse": swap["pixel_mse_0_255"],
                         "swap_over_clean_pixel_mse": swap["pixel_mse_0_255"] / clean["pixel_mse_0_255"]})
    report = {
        "schema_version": 1,
        "status": "completed_native_cached_decode_observation_calibration",
        "research_goal_status": "active_unachieved",
        "previous_goal_turn_classification": "progress: pinned WoG/DiLA source checks and GenSafeNav feasibility findings changed which experiments were qualified; no task experiment in that previous turn",
        "current_goal_turn_classification": "progress: six native VAE decodes add observable-RGB evidence and calibrated reconstruction controls; not a new task-harm experiment",
        "sources": {
            "protocol": "configs/cosmos_cached_decode_gate_20261016.json",
            "launched_protocol": str((RUN / "launched_config.json").relative_to(ROOT)),
            "summary": str((RUN / "summary.json").relative_to(ROOT)),
            "executed_script": str((RUN / "launched_script.py").relative_to(ROOT)),
            "runner": "scripts/probe_cosmos_cached_decode.py",
            "analysis": "scripts/analyze_cosmos_cached_decode.py",
            "latent_and_measured_rgb_source": cfg["source"],
            "previous_value_factorial": "reports/cosmos_future_modality_factorial_decision_20261016.json"
        },
        "scope": cfg["scope"],
        "method": {
            "native_path": "Restore latent slots0/1/4/5, decode all9 packed slots using the native BF16 VAE, use FP32 RGB postprocessing for all conditions, take frames21/25.",
            "truth_calibration": "Restored truth Value input equals joint measured-future VAE encoding in the BF16 prefix consumed by both future images. Placeholder restoration leaves the complete joint truth encoding unchanged.",
            "reference": "Saved raw33 uint8 images already contain native camera flip, JPEG95, resize and center crop. No repeat preprocessing or image editing.",
            "time": "The33 frames are modality packaging, not physical timesteps. Both future cameras correspond to the same endpoint after16 controls.",
            "unit": "RGB errors in0..255 pixel units; PSNR from255^2/MSE. No physical-state or utility interpretation."
        },
        "rows": rows,
        "findings": {
            "all_truth_reconstruction_psnr_db": [min(r["truth_psnr_db"] for r in rows), max(r["truth_psnr_db"] for r in rows)],
            "all_four_camera_candidate_swap_mse_exceeds_clean": all(r["swap_pixel_mse"] > r["clean_pixel_mse"] for r in rows),
            "swap_over_clean_pixel_mse_range": [min(r["swap_over_clean_pixel_mse"] for r in rows), max(r["swap_over_clean_pixel_mse"] for r in rows)],
            "supported": "The fixed Swap discrepancy is visible in decoded observation space and is not merely a numerical difference in unused latent coordinates. Truth reconstruction is a separately measured codec baseline.",
            "not_supported": "No isolation of physical content versus appearance/representation, no calibrated drawer/bowl/robot state estimate from predicted images, and no evidence that these pixel errors specifically mediate the ranking reversal."
        },
        "available_physical_truth": {
            "existing_cache": "Actual per-step EEF pose, gripper joint positions, actions and success/reward are logged. Both16-control endpoints have success=false and reward0.",
            "missing": "No endpoint full simulator state, named drawer joint, bowl pose, contacts or camera calibration is saved. candidate*_states.npz contains only initial/settled states from before the candidate controls.",
            "source": "scripts/replay_cosmos_first_query_truth.py",
            "interpretation": "Gripper joint position is not contact truth; decoded images are not independent simulator state."
        },
        "contact_sheet": summary["contact_sheet"],
        "csv": str(OUT.with_suffix(".csv").relative_to(ROOT)),
        "execution": {
            "activity": summary["activity"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_cuda_reserved_bytes": summary["peak_cuda_reserved_bytes"],
            "dedicated_gpu_requested": False,
            "successful_session": {"session_id": 89946, "exit_code": 0, "terminal_chunk_id": "280d07"},
            "prior_technical_attempt": {"summary": cfg["prior_technical_attempt"], "session_id": 58018,
                "exit_code": 1, "reason": "Missing torchvision in the initially selected environment;0model loads/0decodes.",
                "handling": "Preserved failed attempt; used the original Torch2.2/numpy1.26 environment. All decode input arguments changed to FP32 before any decode was performed, removing a review-detected postprocessing confound."},
            "source_review": "Independent review confirmed native mapping/placeholder restoration and identified the dtype confound before the successful run. No duplicate model execution."
        },
        "other_bounded_feasibility_checks_this_turn": [
            {"topic": "Cosmos content/style countercheck", "result": "No identifiable separation from existing cached latents. Round-trip decoding, filtering or averaging cannot be assumed content-preserving; no Value calls launched."},
            {"topic": "Native private-goal interface", "result": "Existing JEPA PointMaze hides the marker and scores the goal separately, but its sole tested nontrivial task failed. DINO MetaWorld's competent tiny Clean panel is goal-observable. No privacy or normal-task run launched.",
             "source": "runs/jepa_native_microbatch_ep2_v1/summary.json"},
            {"topic": "DINO PointMaze public resource lead", "result": "Official upstream lists a separate reproduced PointMaze checkpoint (about275MB), not currently validated here. Public metadata is not native competence or a new privacy mechanism; no download.",
             "url": "https://github.com/facebookresearch/jepa-wms"},
            {"topic": "World4Scorer multiplicity", "result": "Training samples unsafe candidates and fills small valid pools with replacement; duplicate frequency can encode sampling/support information. Unequal duplication is an added candidate-assembly intervention, not a proven physical-invariance violation. No matched real scene/metric-cache bundle is locally qualified; no model run.",
             "source": "runs/world4scorer_source_screen_20261016/source/navsim/agents/drivoR/drivor_model.py"}
        ],
        "decision": {
            "go": "Reuse the calibrated native image decoder for future independently grounded physical-content tests; retain the prior local Cosmos utility evidence.",
            "no_go": "No new security mechanism, actual harm, effective defense or independent generalization established. The failed30-root expansion gate and all previous stop rules remain in force.",
            "next_decisive_gap": "A calibrated physical readout plus separately controlled content/appearance variables would be needed to attribute the local selection loss to physical forecast error. Large latent or pixel MSE alone is insufficient."
        },
        "reproduce": {
            "table_only": "python scripts/analyze_cosmos_cached_decode.py",
            "decode_to_fresh_directory": ".venv-worldagen/bin/python scripts/probe_cosmos_cached_decode.py --output runs/cosmos_cached_decode_reproduction"
        }
    }
    OUT.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with OUT.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"report": str(OUT.with_suffix('.json').relative_to(ROOT)), "rows": len(rows),
                      "findings": report["findings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
