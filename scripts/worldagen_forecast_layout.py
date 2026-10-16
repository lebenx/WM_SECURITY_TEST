"""WorldAgen decoder layout and patch-normalization contracts."""
import torch


def primary_forecasts(world: torch.Tensor, batch_size: int, image_times: int = 2):
    """[B*T, view, patch, pixel] -> [B, T, patch, pixel], primary view."""
    if world.ndim != 4 or world.shape[:2] != (batch_size * image_times, 2):
        raise ValueError(f"unexpected decoder layout {tuple(world.shape)}")
    return world.reshape(batch_size, image_times, 2, *world.shape[2:])[:, :, 0]


def restore_patch_statistics(tokens: torch.Tensor, reference_processed: torch.Tensor):
    """Use current-frame scalar patch statistics (all pixels AND channels).

    Exact inverse when tokens are the normalized reference patches. For a
    future prediction this is an approximation, never true future statistics.
    """
    grid = tokens.reshape(14, 14, 16, 16, 3)
    ref = reference_processed.to(tokens).reshape(3, 14, 16, 14, 16)
    ref = ref.permute(1, 3, 2, 4, 0)
    dims = (2, 3, 4)
    mean = ref.mean(dim=dims, keepdim=True)
    std = (ref.var(dim=dims, keepdim=True) + 1e-6).sqrt()
    return grid * std + mean
