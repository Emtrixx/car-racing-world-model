import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.vq_conv_vae import VQVAE
from src.world_model import WorldModelGRU


class GruDreamEnv(gym.Env):
    """
    A Gym environment that simulates the CarRacing-v3 environment using a trained World Model.
    The agent interacts with the world model's "dream" instead of the real environment.
    """

    def __init__(self, world_model: WorldModelGRU, vq_vae: VQVAE, device, seed, horizon, real_buffer):
        super(GruDreamEnv, self).__init__()

        self.world_model = world_model
        self.vq_vae = vq_vae
        self.device = device
        self.seed = seed
        self.horizon = horizon
        self.real_buffer = real_buffer

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)

        self.hidden_state = None
        self.current_latent_codes = None
        self.current_step = 0
        self.valid_start_indices = []

    def _find_valid_start_indices(self):
        """
        Scans the real buffer to find all indices that can start a valid dream sequence.
        A valid start is any state that is not a terminal state.
        """
        self.valid_start_indices = [
            i for i, transition in enumerate(self.real_buffer)
            if not transition['done']
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)

        self._find_valid_start_indices()

        if not self.valid_start_indices:
            raise ValueError("No valid start indices found in the replay buffer. "
                             "Buffer might be empty or contain only terminal states.")

        start_idx = random.choice(self.valid_start_indices)
        initial_tokens = self.real_buffer[start_idx]['prev_tokens'].to(self.device)

        # Squeeze to [grid_size, grid_size]
        self.current_latent_codes = initial_tokens.view(self.world_model.grid_size, self.world_model.grid_size)

        # Initialize the GRU hidden state for a batch size of 1
        self.hidden_state = self.world_model.get_initial_hidden_state(batch_size=1, device=self.device)

        obs = self._decode_latent_to_obs(self.current_latent_codes)
        self.current_step = 0
        return obs, {}

    def step(self, action):
        action_tensor = torch.from_numpy(action).unsqueeze(0).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # The GRU model returns a tuple of (logits, rewards, dones, hidden_state, stochastic_dist)
            (
                predicted_latent_logits,
                predicted_reward,
                predicted_done_logits,
                next_hidden_state,
                _,  # We don't need the stochastic distribution here
            ) = self.world_model(
                self.current_latent_codes.flatten().unsqueeze(0).unsqueeze(1),  # Add batch and sequence dimensions
                action_tensor,
                self.hidden_state
            )

        self.hidden_state = next_hidden_state

        # Sample the next latent state from the predicted logits
        predicted_latent_probs = torch.softmax(predicted_latent_logits.view(-1, self.world_model.codebook_size), dim=-1)
        predicted_latent_codes_flat = torch.multinomial(predicted_latent_probs, 1).squeeze()
        self.current_latent_codes = predicted_latent_codes_flat.view(self.world_model.grid_size,
                                                                     self.world_model.grid_size)

        obs = self._decode_latent_to_obs(self.current_latent_codes)
        reward = predicted_reward.item()
        done = torch.sigmoid(predicted_done_logits).item() > 0.5

        self.current_step += 1
        truncated = self.current_step >= self.horizon

        return obs, reward, done, truncated, {}

    def _decode_latent_to_obs(self, latent_codes):
        with torch.no_grad():
            latent_codes = latent_codes.to(self.device)
            quantized = self.vq_vae.vq_layer.embeddings.data[latent_codes.long()]
            quantized = quantized.permute(2, 0, 1).unsqueeze(0)
            decoded_obs = self.vq_vae.decoder(quantized)

        obs_numpy = decoded_obs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return obs_numpy.astype(np.float32)  # Match transformer output

    def render(self, mode='human'):
        pass

    def close(self):
        pass
