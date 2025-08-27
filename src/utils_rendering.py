from typing import List, Tuple

import imageio
import torch

from src.vq_conv_vae import VQVAE
from src.world_model import WorldModelGRU


def get_starting_state_from_sequence(image_paths: List[str],
                                     world_model: WorldModelGRU,
                                     vq_vae: VQVAE, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads a sequence of images, encodes them, and processes them sequentially
    to prime the world model's state.

    Args:
        image_paths (List[str]): A list of paths to the pre-processed sample images, in order.
        world_model (WorldModelGRU): The trained world model.
        vq_vae (VQVAE): The trained VQ-VAE.
        device: The torch device.

    Returns:
        tuple: A tuple containing (final_primed_hidden_state, last_frame_reconstruction, last_tokens).
    """
    print(f"Initializing dream from a sequence of {len(image_paths)} images...")

    last_frame_reconstruction = None
    last_tokens = None

    hidden_state = world_model.get_initial_hidden_state(batch_size=1, device=device)
    with torch.no_grad():
        for image_path in image_paths:
            frame_np = imageio.imread(image_path)
            frame_tensor = torch.tensor(frame_np, dtype=torch.float32, device=device) / 255.0
            # Adjust for grayscale or RGB images
            if len(frame_tensor.shape) == 2:  # Grayscale
                frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dimension
            elif len(frame_tensor.shape) == 3 and frame_tensor.shape[-1] == 3:  # RGB
                frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert HWC to CHW
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

            reconstruction, _, _, indices, _, _ = vq_vae(frame_tensor)
            indices = indices.view(1, 1, -1)  # Reshape for model [B, T, N]
            last_frame_reconstruction = reconstruction
            last_tokens = indices

            dummy_action = torch.zeros(1, 1, world_model.action_embedding.in_features, device=device)
            _, _, _, hidden_state, _ = world_model(
                obs_tokens=indices,
                actions=dummy_action,
                initial_hidden_state=hidden_state
            )
    return hidden_state, last_frame_reconstruction, last_tokens
