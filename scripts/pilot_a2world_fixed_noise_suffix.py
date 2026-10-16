"""Fixed-noise forward probe for temporal nonlocality in A2World.

This deliberately stops before decoding or planning.  It holds the latent
video input, timestep, camera layout, and text context fixed, changes only
the last 16 of the released 20 action steps, and measures the denoiser's
latent output at every video time slot.  A nonzero early-slot change is
carrier evidence for nonlocal action conditioning; it is not by itself a
task-level safety claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "A2World" / "a2world-pretrained.pt"
OUT = ROOT / "runs" / "a2world_fixed_noise_suffix_v0" / "summary.json"


def build_network():
    # Importing the vendored model keeps this probe independent of the gated
    # Cosmos tokenizer.  The denoiser itself and all 826 released tensors are
    # used.  A meta-device construction avoids spending two minutes on random
    # initialization before assigning the checkpoint tensors.
    sys.path.insert(0, str(ROOT / "vendor" / "A2World" / "world_model"))
    from cosmos_predict2.models.text2image_dit import SACConfig
    from cosmos_predict2.models.video2world_multiview_action_dit import ActionConditionedMultiViewDiT

    with torch.device("meta"):
        net = ActionConditionedMultiViewDiT(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            atten_backend="transformer_engine",
            pos_emb_cls="rope3d",
            pos_emb_learnable=False,
            pos_emb_interpolation="crop",
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=0.25,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            sac_config=SACConfig(every_n_blocks=1, mode="predict2_2b_720"),
            state_t=6,
            n_cameras_emb=4,
            view_condition_dim=7,
            concat_view_embedding=True,
            action_dim=280,
            use_black=True,
            use_history=False,
        )
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    release_state = {key[4:]: value for key, value in state.items()}
    # Materialize directly on the target GPU; assigning CPU tensors to a meta
    # module leaves omitted buffers on meta and makes a later `.to()` invalid.
    net = net.to_empty(device="cuda")
    report = net.load_state_dict(release_state, strict=False, assign=True)
    # The release intentionally omits optimizer/statistics and Transformer
    # Engine extra-state entries.  All learned tensor weights are present.
    learned_missing = [key for key in report.missing_keys if not key.endswith("_extra_state") and not key.startswith("accum_")]
    if learned_missing or report.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={learned_missing[:5]}, unexpected={report.unexpected_keys[:5]}")
    return net, len(report.missing_keys), len(report.unexpected_keys)


def main() -> None:
    start = time.time()
    net, missing_count, unexpected_count = build_network()
    device = torch.device("cuda")
    # `assign=True` keeps the mmap-backed CPU storage; now transfer the
    # materialized parameters and buffers to the 4090.
    net = net.to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    latent = torch.randn(1, 16, 18, 4, 4, device=device, dtype=torch.bfloat16)
    condition_mask = torch.zeros(1, 1, 18, 4, 4, device=device, dtype=torch.bfloat16)
    padding_mask = torch.zeros(1, 1, 4, 4, device=device, dtype=torch.bfloat16)
    timestep = torch.full((1, 18), 0.5, device=device, dtype=torch.bfloat16)
    fps = torch.full((1,), 10.0, device=device)
    view_indices = torch.tensor([[0] * 6 + [1] * 6 + [2] * 6], device=device)
    context = torch.zeros(1, 3 * 512, 1024, device=device, dtype=torch.bfloat16)

    base_action = torch.linspace(-0.25, 0.25, 20 * 14, device=device, dtype=torch.bfloat16).reshape(1, 20, 14)
    suffix_action = base_action.clone()
    suffix_action[:, 4:] = torch.flip(suffix_action[:, 4:], dims=[1])
    late_action = base_action.clone()
    late_action[:, 19] = late_action[:, 19] * -1

    with torch.inference_mode():
        base_output = net(
            latent,
            timestep,
            context,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            padding_mask=padding_mask,
            action=base_action,
            view_indices_B_T=view_indices,
        )
        suffix_output = net(
            latent,
            timestep,
            context,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            padding_mask=padding_mask,
            action=suffix_action,
            view_indices_B_T=view_indices,
        )
        late_output = net(
            latent,
            timestep,
            context,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            padding_mask=padding_mask,
            action=late_action,
            view_indices_B_T=view_indices,
        )

    delta = (suffix_output - base_output).float()
    late_delta = (late_output - base_output).float()
    per_slot = delta.abs().mean(dim=(0, 1, 3, 4))
    late_per_slot = late_delta.abs().mean(dim=(0, 1, 3, 4))
    relative = per_slot / base_output.float().abs().mean(dim=(0, 1, 3, 4)).clamp_min(1e-8)
    result = {
        "status": "completed_a2world_fixed_noise_suffix_forward",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "missing_release_extra_state_keys": missing_count,
        "unexpected_keys": unexpected_count,
        "video_latent_shape": list(base_output.shape),
        "fixed_prefix_steps": 4,
        "changed_suffix_steps": 16,
        "per_temporal_latent_mean_abs": per_slot.cpu().tolist(),
        "late_single_step_per_temporal_latent_mean_abs": late_per_slot.cpu().tolist(),
        "per_temporal_latent_relative_mean_abs": relative.cpu().tolist(),
        "prefix_mean_abs": float(per_slot[:6].mean()),
        "suffix_mean_abs": float(per_slot[6:].mean()),
        "exact_zero_prefix": bool(torch.equal(delta[:, :, :6], torch.zeros_like(delta[:, :, :6]))),
        "late_step_exact_zero_prefix": bool(torch.equal(late_delta[:, :, :6], torch.zeros_like(late_delta[:, :, :6]))),
        "late_step_prefix_mean_abs": float(late_per_slot[:6].mean()),
        "late_step_suffix_mean_abs": float(late_per_slot[6:].mean()),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - start,
        "scope": "fixed-noise latent forward; no decoded video or task-level harm",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
