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

    def __init__(self, agent, world_model: WorldModelTransformer, vq_vae: VQVAE, device, seed, history_length, horizon, real_buffer):
        super(DreamEnvTransformer, self).__init__()

        self.agent = agent
        self.world_model = world_model
        self.vq_vae = vq_vae
        self.device = device
        self.seed = seed
        self.history_length = history_length
        self.horizon = horizon
        self.real_buffer = real_buffer

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)

        self.action_history = deque(maxlen=self.history_length)
        self.token_history = deque(maxlen=self.history_length)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)

        # Sample a starting sequence from the real replay buffer
        start_idx = random.randint(0, len(self.real_buffer) - self.history_length - 1)
        
        self.action_history.clear()
        self.token_history.clear()

        for i in range(self.history_length):
            data_point = self.real_buffer[start_idx + i]
            self.action_history.append(data_point['action'])
            self.token_history.append(data_point['prev_tokens'])

        initial_obs_tokens = self.real_buffer[start_idx + self.history_length]['prev_tokens']
        obs = self._decode_latent_to_obs(initial_obs_tokens.view(self.world_model.grid_size, self.world_model.grid_size))
        self.current_step = 0
        
        return obs, {}

    def step(self, action):
        action_tensor = torch.from_numpy(action).float().to(self.device).unsqueeze(0)
        self.action_history.append(action_tensor.squeeze(0).cpu())

        with torch.no_grad():
            action_hist_tensor = torch.stack(list(self.action_history)).to(self.device).unsqueeze(0)
            token_hist_tensor = torch.stack(list(self.token_history)).to(self.device).unsqueeze(0)

            pred_logits, pred_reward, pred_done_logits, _ = self.world_model(action_hist_tensor, token_hist_tensor)

            # Sample the next latent state from the predicted logits
            pred_probs = torch.softmax(pred_logits.view(-1, self.world_model.codebook_size), dim=-1)
            next_tokens_flat = torch.multinomial(pred_probs, 1).squeeze()
            self.token_history.append(next_tokens_flat.cpu())

        obs = self._decode_latent_to_obs(next_tokens_flat.view(self.world_model.grid_size, self.world_model.grid_size))
        reward = pred_reward.item()
        done = torch.sigmoid(pred_done_logits).item() > 0.5
        
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