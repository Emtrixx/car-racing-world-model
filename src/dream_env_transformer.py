# src/dream_env_transformer.py
import random
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from collections import deque

from src.transformer_world_model import WorldModelTransformer
from src.vq_conv_vae import VQVAE


class DreamEnvTransformer(gym.Env):
    """
    A Gym environment that simulates the CarRacing-v3 environment using a trained Transformer World Model.
    The agent interacts with the world model's "dream" instead of the real environment.
    """

    def __init__(self, world_model: WorldModelTransformer, vq_vae: VQVAE, device, seed, history_length, horizon,
                 start_state_pool: list):
        super(DreamEnvTransformer, self).__init__()

        self.world_model = world_model
        self.vq_vae = vq_vae
        self.device = device
        self.seed = seed
        self.history_length = history_length
        self.horizon = horizon
        self.start_state_pool = start_state_pool

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)

        self.action_history = deque(maxlen=self.history_length)
        self.token_history = deque(maxlen=self.history_length)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)

        if not self.start_state_pool:
            raise ValueError("Start state pool is empty. Cannot reset dream environment.")

        # Sample a starting sequence from the pre-computed valid pool
        start_state = random.choice(self.start_state_pool)

        self.action_history.clear()
        self.token_history.clear()

        # Prime the history deques from the chosen start state
        self.action_history.extend(start_state['action_history'])
        self.token_history.extend(start_state['token_history'])

        # The next observation is the one following the history sequence
        initial_obs_tokens = start_state['initial_obs_tokens']
        obs = self._decode_latent_to_obs(
            initial_obs_tokens.view(self.world_model.grid_size, self.world_model.grid_size))
        self.current_step = 0

        return obs, {}

    def step(self, action):
        action_tensor = torch.from_numpy(action).float().to(self.device).unsqueeze(0)
        self.action_history.append(action_tensor.squeeze(0).cpu())

        # Ensure the model is in evaluation mode for efficient inference
        self.world_model.eval()

        # History tensors are expected to be [B, H, ...], so we add a batch dim of 1
        action_hist_tensor = torch.stack(list(self.action_history)).to(self.device).unsqueeze(0)
        token_hist_tensor = torch.stack(list(self.token_history)).to(self.device).unsqueeze(0)

        # Use the efficient generate method for single-step inference
        next_tokens, reward_tensor, done_tensor = self.world_model.generate(action_hist_tensor,
                                                                            token_hist_tensor)

        # next_tokens is [B, num_tokens], reward is [B, 1], done is [B, 1]
        # We remove the batch dimension (which is 1) for processing
        next_tokens_no_batch = next_tokens.squeeze(0)
        self.token_history.append(next_tokens_no_batch.cpu())

        obs = self._decode_latent_to_obs(
            next_tokens_no_batch.view(self.world_model.grid_size, self.world_model.grid_size))
        reward = reward_tensor.item()
        done = done_tensor.item()

        self.current_step += 1
        truncated = self.current_step >= self.horizon

        return obs, reward, done, truncated, {}

    def _decode_latent_to_obs(self, latent_codes):
        with torch.no_grad():
            quantized = self.vq_vae.vq_layer.embeddings.data[latent_codes.long()]
            quantized = quantized.permute(2, 0, 1).unsqueeze(0)
            decoded_obs = self.vq_vae.decoder(quantized)

        obs_numpy = decoded_obs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return obs_numpy.astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
