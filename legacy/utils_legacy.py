import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from utils import ENV_NAME, NUM_STACK, LATENT_DIM, DEVICE, ActionClipWrapper, FrameStackWrapper


class LatentStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, vae_model, transform_fn, latent_dim, num_stack, device):
        super().__init__(env)
        self.vae_model = vae_model.to(device).eval()
        self.transform_fn = transform_fn
        self.latent_dim = latent_dim
        self.num_stack = num_stack
        self.device = device

        # New observation space is the concatenated latent vectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_stack * self.latent_dim,),
            dtype=np.float32
        )

    def observation(self, obs_stack_raw_numpy):
        # obs_stack_raw_numpy comes from FrameStackWrapper: (num_stack, H, W, C) dtype=uint8
        latent_vectors_list = []
        for i in range(obs_stack_raw_numpy.shape[0]):  # Iterate through N frames in the stack
            raw_frame = obs_stack_raw_numpy[i]  # Single raw frame (H, W, C)

            # Apply torchvision transform (expects PIL or HWC NumPy, outputs Tensor C,H,W)
            processed_frame_tensor = self.transform_fn(raw_frame)  # Should be (C, H, W)

            # Add batch dim, move to device for VAE
            processed_frame_tensor = processed_frame_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, _ = self.vae_model.encode(processed_frame_tensor)  # mu shape (1, LATENT_DIM)
                latent_vectors_list.append(mu.squeeze(0))  # Squeeze to (LATENT_DIM), keep on device

        # Concatenate the N latent vectors (still on device)
        concatenated_latents_tensor = torch.cat(latent_vectors_list, dim=0)  # Shape: (num_stack * LATENT_DIM,)

        # Environment observations should be NumPy arrays on CPU
        return concatenated_latents_tensor.cpu().numpy()


def make_env(vae_model_instance,  # The loaded and initialized VAE model
             transform_function,  # torchvision.transforms.Compose object
             env_id=ENV_NAME,
             frame_stack_num=NUM_STACK,
             render_mode=None,
             gamma=0.99,
             single_latent_dim=LATENT_DIM,
             device_for_vae=DEVICE,
             max_episode_steps=None,
             ):
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = ActionClipWrapper(env)  # Clip actions from agent before they hit the core env logic
    env = FrameStackWrapper(env, frame_stack_num)  # add frame stacking to observation
    env = LatentStateWrapper(env, vae_model_instance, transform_function,
                             single_latent_dim, frame_stack_num, device_for_vae)
    # env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -1.0, 1.0)) # clip reward
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # normalize reward

    print(f"Final wrapped environment observation space: {env.observation_space}")
    print(f"Sample observation shape from final env: {env.observation_space.sample().shape}")

    return env
