# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Pure data codecs copied unchanged from NVlabs/cosmos-policy.
Revision: 18a2accadf4e7a3531e56754102af5a24d2316da.
Source module and original SHA256 appear before each extracted group.
"""
from __future__ import annotations
import io
from typing import List, Any, Dict
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
COSMOS_IMAGE_SIZE = 224

# Original source: cosmos_policy/datasets/dataset_utils.py
# SHA256: 0a6b245f0ca3c1af8d657e7b5351d9fc5fc267ddde31d54df4befb811622c3f9

def apply_jpeg_compression_np(image_np: np.ndarray, quality: int = 95) -> np.ndarray:
    """Apply JPEG compression/decompression to a NumPy image or batch of images.

    Accepts either a single image with shape (H, W, C) **or** a batch of images
    with shape (B, H, W, C). All inputs must be uint8 RGB.

    Args:
        image_np (np.ndarray): Input image(s) as uint8 array(s).
        quality (int): JPEG quality factor (1–95).

    Returns:
        np.ndarray: JPEG-compressed (and re-decoded) image(s) with the same shape
        as the input.
    """

    def _compress_single(img: np.ndarray) -> np.ndarray:
        """JPEG-compress a single image (H, W, C)."""
        assert img.dtype == np.uint8, f"Expected uint8 image but got {img.dtype}"
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return np.array(compressed_img)

    # Determine if image_np is a single image or a batch
    if image_np.ndim == 3:
        # Single image
        return _compress_single(image_np)
    elif image_np.ndim == 4:
        # Batch of images: iterate and stack
        compressed = [_compress_single(img) for img in image_np]
        return np.stack(compressed, axis=0)
    else:
        raise ValueError(f"Expected image_np with shape (H, W, C) or (B, H, W, C) but got shape {image_np.shape}")

def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes multiple images to some target size.

    Assumes that the resulting images will be square.

    Args:
        images (np.ndarray): Input images with shape (T, H, W, C)
        target_size (int): Target image size (square)

    Returns:
        np.ndarray: Resized images with shape (T, target_size, target_size, C)
    """
    assert len(images.shape) == 4, f"Expected 4 dimensions in images but got: {len(images.shape)}"

    # If the images are already the target size, return them as is
    if images.shape[-3:] == (target_size, target_size, 3):
        return images.copy()

    # Get the number of images
    num_images = images.shape[0]

    # Create an empty array for the resized images
    # We assume the channel dimension C remains the same
    C = images.shape[3]
    resized_images = np.empty((num_images, target_size, target_size, C), dtype=images.dtype)

    # Resize each image
    for i in range(num_images):
        resized_images[i] = np.array(Image.fromarray(images[i]).resize((target_size, target_size)))

    return resized_images

# Original source: cosmos_policy/experiments/robot/cosmos_utils.py
# SHA256: a95c74f33f519f4c0ce048c3ba2694989d3a7c431bf0f324a274413cde4c0b47

def check_images_format(images: np.ndarray) -> None:
    """
    Validate input images format.

    Args:
        images (np.ndarray): Images to check

    Returns:
        None

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(images, np.ndarray)
    has_correct_shape = len(images.shape) == 4 and images.shape[-1] == 3
    has_correct_dtype = images.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect images format detected! Make sure that the input images are a "
        "numpy array with shape (T, H, W, 3) and dtype np.uint8!"
    )

def apply_image_transforms(images: np.ndarray) -> np.ndarray:
    """
    Apply test-time image transformations to match the image augmentations used at train time.

    At train time, we use random resized crops (90% area) and color jitter. At test time, we only need to
    do a 90% area center crop.

    Args:
        images (np.ndarray): Images of shape (T, H, W, C) and dtype np.uint8

    Returns:
        np.ndarray: Images with test-time transformations applied
    """
    # Get dimensions
    _, H, W, C = images.shape
    assert H == W, f"Image height and width must be equal! Got H == {H} and W == {W}"
    assert C == 3, f"Number of channels should be 3! Got C == {C}"
    # Convert numpy array to torch tensor and reshape to (T, C, H, W)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    # Apply deterministic transformations to all images
    results = []
    for img in images_tensor:
        # Apply center crop. Since we used 90% area in training, use a 90% center crop here
        crop_size = int(H * 0.9**0.5)  # Square root because (sqrt(.9) * H) * (sqrt(.9) * H) = 0.9 * H^2
        img_crop = F.center_crop(img, crop_size)
        # Resize back to original size
        img_resized = F.resize(img_crop, [H, W], antialias=True)
        results.append(img_resized)
    # Combine results
    transformed_images = torch.stack(results)
    # Convert back to numpy array with shape (T, H, W, C)
    transformed_images = transformed_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return transformed_images

def prepare_images_for_model(images: List[np.ndarray], cfg, flip_images: bool = False) -> List[Image.Image]:
    """
    Prepare images for Cosmos model input by resizing and cropping as needed.

    Args:
        images (List[np.ndarray]): List of input images as numpy arrays
        cfg: Configuration object with parameters
        flip_images (bool): Whether to flip images vertically across x-axis

    Returns:
        np.ndarray: Processed images ready for the model
    """
    images = np.stack(images, axis=0)  # (T, H, W, C)
    # Check that the images have the right format
    check_images_format(images)
    # Flip images vertically across x-axis if needed (e.g., for LIBERO and RoboCasa)
    if flip_images:
        images = np.flipud(images)
    # Apply JPEG compression (if trained on JPEG-compressed images)
    if cfg.use_jpeg_compression:
        images = apply_jpeg_compression_np(images, quality=95)
    # Resize images to match training distribution
    processed_images = resize_images(images, COSMOS_IMAGE_SIZE)
    # Apply image transformations if trained with image augmentations
    if cfg.trained_with_image_aug:
        processed_images = apply_image_transforms(processed_images)
    return processed_images

def extract_action_chunk_from_latent_sequence(
    output_latent: torch.Tensor, action_shape: tuple, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract the predicted action chunk from the generated latent sequence.

    Args:
        output_latent (torch.Tensor): The full latent sequence generated by the model, with shape (B, C', T', H', W')
        action_shape (tuple): Target shape of the action chunk: (chunk_size, action_dim)
        action_indices (torch.Tensor): Batch indices specifying which index of the latent sequence contains the action

    Returns:
        torch.Tensor: Batch of extracted action chunks, with shape (B, chunk_size, action_dim)
    """
    # Get the action latent frame
    batch_indices = torch.arange(output_latent.shape[0], device=output_latent.device)
    action_latent_frame = output_latent[batch_indices, :, action_indices, :, :]  # (B, C', H', W')
    # Get shape of latent frames
    batch_size, latent_channels, latent_h, latent_w = action_latent_frame.shape
    # Flatten the action latent frame into a vector (preserving batch dimension)
    flat_action_latent = action_latent_frame.reshape(batch_size, -1)
    num_latent_elements = flat_action_latent.shape[1]
    # Calculate the number of elements in the target action chunk
    assert len(action_shape) == 2, "action_shape should have 2 elements: (chunk_size, action_dim)!"
    num_action_elements = action_shape[0] * action_shape[1]
    # Check that there are enough elements in the latent to extract the action
    assert num_action_elements <= num_latent_elements, (
        f"Action shape {action_shape} requires {num_action_elements} elements, but the latent only has {num_latent_elements} elements!"
    )
    # Calculate how many full action chunks we can extract
    num_action_chunks = num_latent_elements // num_action_elements
    # Get the full action chunks in the flat latent and then reshape to separate out the chunks
    # New shape: (batch_size, num_action_chunks, num_action_elements)
    all_action_chunks = flat_action_latent[:, : num_action_chunks * num_action_elements].reshape(
        batch_size, num_action_chunks, num_action_elements
    )
    # Reshape each chunk to the target shape
    # New shape: (batch_size, num_action_chunks, chunk_size, action_dim])
    all_action_chunks = all_action_chunks.reshape(batch_size, num_action_chunks, action_shape[0], action_shape[1])
    # Take the average over all chunks, along dimension 1 (the num_action_chunks dimension)
    # New shape: (batch_size, chunk_size, action_dim)
    final_action_chunk = torch.mean(all_action_chunks, dim=1)
    return final_action_chunk

def extract_value_from_latent_sequence(output_latent: torch.Tensor, value_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract the predicted value from the generated latent sequence.

    Args:
        output_latent (torch.Tensor): The full latent sequence generated by the model, with shape (B, C', T', H', W')
        value_indices (torch.Tensor): Batch indices specifying which index of the latent sequence contains the value

    Returns:
        torch.Tensor: Batch of extracted values, with shape (B, value_dim)
    """
    # Get the value latent frame
    batch_indices = torch.arange(output_latent.shape[0], device=output_latent.device)
    value_latent_frame = output_latent[batch_indices, :, value_indices, :, :]  # (B, C', H', W')
    # Flatten value latent frame into a vector (B, C' * H' * W')
    flat_value_latent = value_latent_frame.reshape(value_latent_frame.shape[0], -1)
    # Take the average across all elements in the value latent frame
    final_value = torch.mean(flat_value_latent, dim=1)
    return final_value

def unnormalize_actions(actions: np.ndarray, dataset_stats: dict, scale_multiplier: float = 1.0):
    """
    Unnormalize actions to the original dataset scale.

    This undoes the normalization used at train time.

    Args:
        actions (np.ndarray): Actions to be unnormalized
        dataset_stats (dict): Dataset statistics needed for the rescaling formula
        scale_multiplier (float): Multiplier to adjust scale from [-scale_multiplier,+scale_multiplier] back to [-1,+1]

    Returns:
        np.ndarray: Unnormalized actions
    """
    actions_min = dataset_stats["actions_min"]
    actions_max = dataset_stats["actions_max"]
    # Reshape actions from (B, chunk_size, action_dim) to (B * chunk_size, action_dim)
    original_shape = actions.shape
    actions = actions.reshape(-1, actions_min.shape[0])
    # First, undo the scale_multiplier scaling
    actions = actions / scale_multiplier
    # Then, scale back to original data scale: x_new = 0.5 * (x + 1) * (x_max - x_min) + x_min
    actions = 0.5 * (actions + 1) * (actions_max - actions_min) + actions_min
    # Reshape actions back to (B, chunk_size, action_dim)
    actions = actions.reshape(original_shape)
    return actions

def rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0):
    """
    Rescale (normalize) proprio to the range [-1,+1] or [0,+1], with optional scaling by scale_multiplier.

    Args:
        proprio (np.ndarray): Proprio to be rescaled
        dataset_stats (dict): Dataset statistics needed for rescaling formula
        non_negative_only (bool): Whether to use [0,+1] range (True) or [-1,+1] range (False)
        scale_multiplier (float): Multiplier to adjust final scale

    Returns:
        np.ndarray: Rescaled proprio
    """
    arr = proprio
    curr_min = dataset_stats["proprio_min"]
    curr_max = dataset_stats["proprio_max"]
    # First, scale to [-1,+1] or [0,+1]:
    # - For [-1,+1]: x_new = 2 * ((x - curr_min) / (curr_max - curr_min)) - 1
    # - For [0,+1]: x_new = (x - curr_min) / (curr_max - curr_min)
    if not non_negative_only:  # [-1,+1]
        rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
    else:  # [0,+1]
        rescaled_arr = (arr - curr_min) / (curr_max - curr_min)
    # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
    rescaled_arr = scale_multiplier * rescaled_arr
    proprio = rescaled_arr
    return proprio

# Original source: cosmos_policy/models/policy_text2world_model.py
# SHA256: d2d6af812e2241fa0d3d8c68620447bd0f23d236412ada9bad5c849df5b848f5

def replace_latent_with_action_chunk(
    x0: torch.Tensor, action_chunk: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replaces the image latent (at the specified action index) in clean input image latents x0 with the action chunk.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and action_chunk has shape (B=32, chunk_size=8, action_dim=7).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,action_indices,:,:] with the action chunk,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        action_chunk (torch.Tensor): Ground-truth action chunk.
        action_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    action_image_latent = x0[batch_indices, :, action_indices, :, :]

    # Create a new tensor with the same shape as action_image_latent, filled with zeros
    result = torch.zeros_like(action_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = action_image_latent.shape

    # Flatten action_chunk (preserving batch dimension)
    flat_action = action_chunk.reshape(batch_size, -1)
    num_action_elements = flat_action.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the action elements
    assert num_action_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full action chunk: {num_action_elements} action elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the action tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_action_elements - 1) // num_action_elements

    # Repeat the action tensor along dimension 1
    repeated_action = flat_action.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_action = repeated_action[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the action chunk values into the beginning of the flattened result
    flat_result[:, :] = repeated_action

    # Reshape back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, action_indices, :, :] = result

    return new_x0

def replace_latent_with_proprio(x0: torch.Tensor, proprio: torch.Tensor, proprio_indices: torch.Tensor) -> torch.Tensor:
    """
    Replaces the image latent (at the specified proprio index) in clean input image latents x0 with the proprio.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and proprio has shape (B=32, proprio_dim=9).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,proprio_indices,:,:] with the proprio,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        proprio (torch.Tensor): Ground-truth proprio.
        proprio_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    proprio_image_latent = x0[batch_indices, :, proprio_indices, :, :]

    # Create a new tensor with the same shape as proprio_image_latent, filled with zeros
    result = torch.zeros_like(proprio_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = proprio_image_latent.shape

    # Get number of proprio elements
    num_proprio_elements = proprio.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the proprio elements
    assert num_proprio_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full proprio: {num_proprio_elements} proprio elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the proprio tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_proprio_elements - 1) // num_proprio_elements

    # Repeat the proprio tensor along dimension 1
    repeated_proprio = proprio.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_proprio = repeated_proprio[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the proprio values into the beginning of the flattened result
    flat_result[:, :] = repeated_proprio

    # Reshape latent back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, proprio_indices, :, :] = result

    return new_x0

