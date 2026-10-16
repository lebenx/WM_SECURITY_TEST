"""CPU placement for the unused CLIP visual branch of native WorldAgen.

Native WorldAgen encodes pixels with MAE and calls only CLIP.encode_text.
CLIP.encode_text reads visual.conv1.weight.dtype, so the dtype must survive.
This helper changes only storage placement, retaining the complete state dict.
It is NOT suitable for the independent planners that call CLIP.encode_image.

Call AFTER the last model.to(device), including the native wrapper constructor.
The guard detects later whole-model moves and accidental visual computation.
This saves steady-state storage, not the peak of building the entire model on
CUDA. No CUDA allocator calls or computation/dtype changes are performed here.
"""
from __future__ import annotations

import torch


def _tensors(module):
    return list(module.named_parameters()) + list(module.named_buffers())


def _storage_id(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _unique_storage_bytes(tensors):
    storage = {}
    for _, tensor in tensors:
        storage[_storage_id(tensor)] = tensor.untyped_storage().nbytes()
    return sum(storage.values())


def _reject_visual_forward(_module, _inputs):
    raise RuntimeError(
        "CPU-offloaded CLIP visual was invoked. This optimization is valid "
        "only for native WorldAgen policy/forecast/loss, not CLIP image scorers."
    )


def _check_visual_placement(model, _inputs):
    if any(value.device.type != "cpu" for _, value in _tensors(model.clip_model.visual)):
        raise RuntimeError(
            "A later model.to(device) moved the unused CLIP visual back. "
            "Call offload_unused_clip_visual after the native wrapper constructor."
        )


def offload_unused_clip_visual(model) -> dict:
    """Move only frozen CLIP.visual to CPU and attach fail-fast usage guards.

    All parameters, buffers, module registration and state-dict keys remain.
    Applying this function repeatedly is supported. Active tensor storage is
    checked for aliasing before the move and unchanged placement afterwards.
    """
    if not hasattr(model, "vision_encoder") or not hasattr(model, "clip_model"):
        raise TypeError("Expected native WorldAgen with separate MAE and CLIP encoders")
    visual = model.clip_model.visual
    before = _tensors(visual)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in visual.parameters()):
        raise ValueError("CLIP visual must be frozen with no outstanding gradients")
    if any(value.device.type == "meta" for _, value in before):
        raise ValueError("Cannot offload a meta tensor")
    prefix = "clip_model.visual."
    active = [(name, value) for name, value in _tensors(model) if not name.startswith(prefix)]
    visual_storage = {_storage_id(value) for _, value in before}
    if any(_storage_id(value) in visual_storage for _, value in active):
        raise ValueError("CLIP visual shares storage with an active branch")
    active_metadata = [(name, id(value), _storage_id(value), value.dtype) for name, value in active]
    state_keys = tuple(model.state_dict())
    dtype = model.clip_model.dtype
    report = {
        "scope": "native WorldAgen policy/forecast/forecast-loss only",
        "parameter_count": sum(value.numel() for value in visual.parameters()),
        "buffer_count": sum(value.numel() for value in visual.buffers()),
        "visual_storage_bytes": _unique_storage_bytes(before),
        "cuda_storage_bytes_before": _unique_storage_bytes(
            [(name, value) for name, value in before if value.device.type == "cuda"]),
        "visual_dtypes": sorted({str(value.dtype) for _, value in before}),
        "devices_before": sorted({str(value.device) for _, value in before}),
    }
    visual.to(device="cpu")  # Deliberately preserve dtype and every weight.
    active_after = [(name, id(value), _storage_id(value), value.dtype)
                    for name, value in _tensors(model) if not name.startswith(prefix)]
    if active_after != active_metadata or tuple(model.state_dict()) != state_keys:
        raise RuntimeError("Offload unexpectedly changed an active tensor or state-dict keys")
    if model.clip_model.dtype != dtype:
        raise RuntimeError("CLIP text dtype changed during visual offload")
    if not hasattr(visual, "_goal_bus_unused_visual_guard"):
        visual._goal_bus_unused_visual_guard = visual.register_forward_pre_hook(_reject_visual_forward)
    if not hasattr(model, "_goal_bus_visual_placement_guard"):
        model._goal_bus_visual_placement_guard = model.register_forward_pre_hook(_check_visual_placement)
    report.update({"devices_after": sorted({str(value.device) for _, value in _tensors(visual)}),
                   "state_dict_keys_preserved": True, "active_tensor_storage_unchanged": True,
                   "clip_text_dtype_preserved": True, "visual_forward_guard": True,
                   "whole_model_move_guard": True})
    return report


_MAE_DECODER_MODULES = ("decoder_embed", "decoder_blocks", "decoder_norm", "decoder_pred")
_MAE_DECODER_PARAMETERS = ("mask_token", "decoder_pos_embed")


def _mae_decoder_tensors(model):
    encoder = model.vision_encoder
    result = [(name, getattr(encoder, name)) for name in _MAE_DECODER_PARAMETERS]
    for name in _MAE_DECODER_MODULES:
        result.extend((name + "." + key, value) for key, value in _tensors(getattr(encoder, name)))
    return result


def _reject_mae_decoder_forward(_module, _inputs):
    raise RuntimeError("CPU-offloaded MAE decoder was invoked; native WorldAgen must call forward_encoder only")


def _check_mae_decoder_placement(model, _inputs):
    if any(value.device.type != "cpu" for _, value in _mae_decoder_tensors(model)):
        raise RuntimeError("A later model.to(device) moved the unused MAE decoder back; reapply offload after wrapper construction")


def offload_unused_mae_decoder(model) -> dict:
    """Move only MAE's unused reconstruction decoder; keep its encoder intact.

    This does NOT move WorldAgen's own image_decoder, mask_token, or position
    embedding, all of which are active in the world forecast. The moved names
    are strictly within vision_encoder and unused by its forward_encoder.
    """
    encoder = model.vision_encoder
    before = _mae_decoder_tensors(model)
    if any(value.requires_grad or value.grad is not None for _, value in before):
        raise ValueError("MAE decoder must be frozen with no outstanding gradients")
    if any(value.device.type == "meta" for _, value in before):
        raise ValueError("Cannot offload a meta tensor")
    names = {"vision_encoder." + name for name, _ in before}
    active = [(name, value) for name, value in _tensors(model) if name not in names]
    unused_storage = {_storage_id(value) for _, value in before}
    if any(_storage_id(value) in unused_storage for _, value in active):
        raise ValueError("MAE decoder shares storage with an active branch")
    metadata = [(name, id(value), _storage_id(value), value.dtype) for name, value in active]
    state_keys = tuple(model.state_dict())
    report = {
        "scope": "unused vision_encoder decoder only; native MAE forward_encoder remains active",
        "parameter_count": sum(value.numel() for _, value in before),
        "unused_storage_bytes": _unique_storage_bytes(before),
        "cuda_storage_bytes_before": _unique_storage_bytes(
            [(name, value) for name, value in before if value.device.type == "cuda"]),
        "devices_before": sorted({str(value.device) for _, value in before}),
        "per_component_bytes": {},
    }
    for name in _MAE_DECODER_MODULES:
        module = getattr(encoder, name)
        report["per_component_bytes"][name] = _unique_storage_bytes(_tensors(module))
        module.to(device="cpu")
        # Guard all descendants: ModuleList itself is not called by the decoder.
        for child in module.modules():
            if not hasattr(child, "_goal_bus_unused_mae_decoder_guard"):
                child._goal_bus_unused_mae_decoder_guard = child.register_forward_pre_hook(_reject_mae_decoder_forward)
    for name in _MAE_DECODER_PARAMETERS:
        value = getattr(encoder, name)
        report["per_component_bytes"][name] = value.untyped_storage().nbytes()
        value.data = value.detach().to(device="cpu")
    active_after = [(name, id(value), _storage_id(value), value.dtype)
                    for name, value in _tensors(model) if name not in names]
    if active_after != metadata or tuple(model.state_dict()) != state_keys:
        raise RuntimeError("MAE offload changed an active tensor or state-dict keys")
    if not hasattr(model, "_goal_bus_mae_decoder_placement_guard"):
        model._goal_bus_mae_decoder_placement_guard = model.register_forward_pre_hook(_check_mae_decoder_placement)
    report.update({"devices_after": sorted({str(value.device) for _, value in _mae_decoder_tensors(model)}),
                   "state_dict_keys_preserved": True, "active_tensor_storage_unchanged": True,
                   "decoder_forward_guard": True, "whole_model_move_guard": True})
    return report


def offload_unused_worldagen_branches(model) -> dict:
    """Combined helper for native policy/forecast/TTT, after model.to()."""
    clip_report = offload_unused_clip_visual(model)
    mae_report = offload_unused_mae_decoder(model)
    return {"clip_visual": clip_report, "mae_decoder": mae_report,
            "unused_storage_bytes": clip_report["visual_storage_bytes"] + mae_report["unused_storage_bytes"],
            "cuda_storage_bytes_before": clip_report["cuda_storage_bytes_before"] + mae_report["cuda_storage_bytes_before"]}
