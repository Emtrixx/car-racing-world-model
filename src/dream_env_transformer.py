import collections

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.transformer_world_model import WorldModelTransformer, HISTORY_LEN
from src.utils import DEVICE
from src.vq_conv_vae import VQVAE


class TransformerDreamEnv(gym.Env):
    """
    A Gym environment that simulates the CarRacing-v3 environment using a trained
    Transformer-based World Model. The agent interacts with the world model's "dream".
    """

    def __init__(
            self,
            world_model: WorldModelTransformer,
            vq_vae: VQVAE,
            initial_frame: np.ndarray,
            history_len: int = HISTORY_LEN
    ):
        super(TransformerDreamEnv, self).__init__()

        self.world_model = world_model
        self.vq_vae = vq_vae
        self.initial_frame = initial_frame
        self.history_len = history_len

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(world_model.action_dim,), dtype=np.float32)
        # Observation is a 64x64 RGB image
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # Buffers for storing the history of latent codes and actions
        self.latent_history = collections.deque(maxlen=self.history_len)
        self.action_history = collections.deque(maxlen=self.history_len)

        # The current latent state (most recent)
        self.current_latent_codes = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Preprocess the initial frame and encode it to get the first latent codes
        # preprocessed_frame = preprocess_observation(self.initial_frame)
        preprocessed_frame = self.initial_frame  # initial_frame is already preprocessed
        obs_tensor = torch.from_numpy(preprocessed_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Encode the initial frame to get latent codes
            _, _, _, encoding_indices, _, _ = self.vq_vae(obs_tensor)
            # Shape: [1, grid_size, grid_size] -> [grid_size * grid_size]
            initial_latent_codes = encoding_indices.view(-1)

        # Clear and populate the history buffers
        self.latent_history.clear()
        self.action_history.clear()

        # Define a zero action for padding the history
        zero_action = torch.zeros(self.action_space.shape, device=DEVICE)

        # Populate the history by repeating the initial state and a zero action
        for _ in range(self.history_len):
            self.latent_history.append(initial_latent_codes)
            self.action_history.append(zero_action)

        # Set the most recent latent state
        self.current_latent_codes = initial_latent_codes

        # Decode the initial latent state to get the first observation
        obs = self._decode_latent_to_obs(self.current_latent_codes)
        return obs, {}

    def step(self, action: np.ndarray):
        # Convert numpy action to a tensor
        action_tensor = torch.from_numpy(action).to(DEVICE)

        # Prepare the history tensors for the model
        # Stack the deques to create tensors of shape [1, history_len, ...]
        latent_hist_tensor = torch.stack(list(self.latent_history), dim=0).unsqueeze(0)
        action_hist_tensor = torch.stack(list(self.action_history), dim=0).unsqueeze(0)

        with torch.no_grad():
            # The transformer predicts the next state based on the entire history
            (
                _,  # predicted_latent_logits (not needed for stepping)
                predicted_reward_tensor,
                predicted_done_logits,
                generated_tokens_indices,  # The predicted next latent state
            ) = self.world_model(action_hist_tensor, latent_hist_tensor)

        # Update the current latent state with the prediction
        # Shape: [1, num_tokens] -> [num_tokens]
        self.current_latent_codes = generated_tokens_indices.squeeze(0)

        # Update the history buffers with the new action and the predicted state
        self.action_history.append(action_tensor)
        self.latent_history.append(self.current_latent_codes)

        # Decode the new latent state to get the next observation
        obs = self._decode_latent_to_obs(self.current_latent_codes)

        # Get the scalar reward and done values
        reward = predicted_reward_tensor.item()
        # Apply sigmoid to done logits and check against a threshold
        done = torch.sigmoid(predicted_done_logits).item() > 0.5
        truncated = False  # This environment does not have a fixed time limit

        return obs, reward, done, truncated, {}

    def _decode_latent_to_obs(self, latent_codes: torch.Tensor) -> np.ndarray:
        """
        Decodes a tensor of latent codes into a NumPy image observation.
        """
        with torch.no_grad():
            # Get the corresponding embeddings from the codebook
            # latent_codes shape: [num_tokens] -> quantized shape: [num_tokens, embed_dim]
            quantized = self.vq_vae.vq_layer.embeddings.data[latent_codes.long()]

            # Reshape to the grid structure expected by the decoder
            # [H*W, C] -> [1, H, W, C] -> [1, C, H, W]
            grid_size = self.world_model.grid_size
            quantized = quantized.view(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)

            # Decode the quantized latents to an image
            decoded_obs_tensor = self.vq_vae.decoder(quantized)

        # Convert the tensor to a NumPy array in the correct format for Gym (H, W, C)
        obs_numpy = decoded_obs_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Convert from [0, 1] float to [0, 255] uint8
        return (obs_numpy * 255).astype(np.uint8)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
