import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.utils import (
    DEVICE,
    preprocess_observation,
    VQ_VAE_CHECKPOINT_BEST_FILENAME,
    WM_CHECKPOINT_FILENAME_GRU,
)
from src.vq_conv_vae import VQVAE
from src.world_model import WorldModelGRU


class GruDreamEnv(gym.Env):
    """
    A Gym environment that simulates the CarRacing-v3 environment using a trained World Model.
    The agent interacts with the world model's "dream" instead of the real environment.
    """

    def __init__(self, world_model: WorldModelGRU, vq_vae: VQVAE, initial_frame: np.ndarray):
        super(GruDreamEnv, self).__init__()

        self.world_model = world_model
        self.vq_vae = vq_vae
        self.initial_frame = initial_frame

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        self.hidden_state = None
        self.current_latent_codes = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # preprocessed_frame = preprocess_observation(self.initial_frame)
        preprocessed_frame = self.initial_frame  # initial_frame is already preprocessed
        obs_tensor = torch.from_numpy(preprocessed_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, _, _, encoding_indices, _, _ = self.vq_vae(obs_tensor)

        # Squeeze to [grid_size, grid_size]
        self.current_latent_codes = encoding_indices.squeeze(0)
        # Flatten to [num_tokens] and add batch dim to get [1, num_tokens]
        tokens_for_encoding = self.current_latent_codes.flatten().unsqueeze(0)

        self.hidden_state = self.world_model.get_initial_hidden_state(1, DEVICE)
        self.hidden_state = self.world_model.encode_observation(tokens_for_encoding,
                                                                self.hidden_state)

        obs = self._decode_latent_to_obs(self.current_latent_codes)
        return obs, {}

    def step(self, action):
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            (
                predicted_latent_logits,
                predicted_reward,
                predicted_done_logits,
                next_hidden_state,
            ) = self.world_model(action_tensor, self.hidden_state)

        self.hidden_state = next_hidden_state

        # Sample the next latent state from the predicted logits
        predicted_latent_probs = torch.softmax(predicted_latent_logits.view(-1, self.world_model.codebook_size), dim=-1)
        predicted_latent_codes_flat = torch.multinomial(predicted_latent_probs, 1).squeeze()
        self.current_latent_codes = predicted_latent_codes_flat.view(self.world_model.grid_size,
                                                                     self.world_model.grid_size)

        obs = self._decode_latent_to_obs(self.current_latent_codes)
        reward = predicted_reward.item()
        done = torch.sigmoid(predicted_done_logits).item() > 0.5
        truncated = False  # This environment does not have a time limit

        return obs, reward, done, truncated, {}

    def _decode_latent_to_obs(self, latent_codes):
        with torch.no_grad():
            quantized = self.vq_vae.vq_layer.embeddings.data[latent_codes.long()]
            quantized = quantized.permute(2, 0, 1).unsqueeze(0)
            decoded_obs = self.vq_vae.decoder(quantized)

        obs_numpy = decoded_obs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (obs_numpy * 255).astype(np.uint8)

    def render(self, mode='human'):
        # Could add rendering logic from play_in_dream here
        pass

    def close(self):
        pass
