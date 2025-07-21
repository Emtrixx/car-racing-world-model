import multiprocessing as mp
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from src.play_game_sb3 import SB3_MODEL_PATH
from src.utils import make_env_sb3, NUM_STACK, VQ_VAE_CHECKPOINT_FILENAME
from src.vq_conv_vae import VQVAE


class WorldModelDataCollector:
    """Collects training data for the World Model by running a pretrained policy."""

    def __init__(self, env, ppo_agent, vq_vae_model, device):
        self.env = env
        self.ppo_agent = ppo_agent
        self.vq_vae_model = vq_vae_model
        self.device = device
        self.replay_buffer = deque(maxlen=2_000_000)  # Increased buffer size

    def get_vq_indices(self, obs_raw_numpy: np.ndarray) -> torch.Tensor:
        """
        Helper function to preprocess a raw observation and get VQ-VAE token indices.

        Args:
            obs_raw_numpy (np.ndarray): A single raw frame from the environment.

        Returns:
            torch.Tensor: A tensor of token indices with shape [1, 16].
        """
        # Preprocess the raw observation
        # processed_frame = preprocess_observation(obs_raw_numpy)
        # env now returns preprocessed frames directly
        processed_frame = obs_raw_numpy

        # Convert to tensor, add batch dim, and send to device
        processed_tensor = torch.tensor(processed_frame, dtype=torch.float32, device=self.device)
        processed_tensor = processed_tensor.permute(2, 0, 1)  # to CHW
        processed_tensor = processed_tensor.unsqueeze(0)  # to BCHW

        # Encode to get the token indices from the VQ-VAE
        with torch.no_grad():
            z_continuous = self.vq_vae_model.encoder(processed_tensor)
            z_continuous = self.vq_vae_model._pre_vq_conv(z_continuous)
            _loss, _quantized_out, _perplexity, encoding_indices_out = self.vq_vae_model.vq_layer(z_continuous)

        return encoding_indices_out.view(1, -1)  # Flatten to [1, 16]

    def collect_steps(self, num_steps: int):
        """Runs the PPO agent in the environment for a given number of steps."""
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()

        # We need the initial state's tokens to start the buffer correctly.
        initial_tokens = self.get_vq_indices(obs[-1])

        for step in tqdm(range(num_steps), desc="Collecting Steps"):
            action, _ = self.ppo_agent.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            next_state_tokens = self.get_vq_indices(next_obs[-1])

            self.replay_buffer.append({
                "prev_tokens": initial_tokens.squeeze(0).to(torch.int64),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done or truncated], dtype=torch.float32),
                "next_tokens": next_state_tokens.squeeze(0).to(torch.int64)
            })

            initial_tokens = next_state_tokens
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                initial_tokens = self.get_vq_indices(obs[-1])


def collect_sequences_worker(worker_id, num_steps_to_collect, env_name, device_str, max_episode_steps):
    """Worker function for parallel data collection."""
    try:
        print(f"[Worker {worker_id}] Starting...")
        env = make_env_sb3(env_id=env_name, frame_stack_num=NUM_STACK, max_episode_steps=max_episode_steps)
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str, env=env)
        vq_vae = VQVAE().to(device_str)
        vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str))
        vq_vae.eval()

        collector = WorldModelDataCollector(env, ppo_agent, vq_vae, device_str)
        collector.collect_steps(num_steps=num_steps_to_collect)
        env.close()

        print(f"[Worker {worker_id}] Finished. Collected {len(collector.replay_buffer)} transitions.")

        # Package data for returning
        data_to_save = {
            'prev_tokens': torch.stack([s['prev_tokens'] for s in collector.replay_buffer]),
            'actions': torch.stack([s['action'] for s in collector.replay_buffer]),
            'rewards': torch.stack([s['reward'] for s in collector.replay_buffer]),
            'dones': torch.stack([s['done'] for s in collector.replay_buffer]),
            'next_tokens': torch.stack([s['next_tokens'] for s in collector.replay_buffer]),
        }

        temp_dir = Path("./tmp_worker_data")
        temp_dir.mkdir(exist_ok=True)
        filepath = temp_dir / f"temp_data_{worker_id}_{int(time.time() * 1000)}.pt"
        torch.save(data_to_save, filepath)
        return str(filepath)

    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_data_parallel(num_steps, device, num_workers, env_name, max_episode_steps):
    """Manages parallel data collection and aggregates results."""
    print(f"Starting parallel collection with {num_workers} workers for {num_steps} total steps...")
    if num_workers <= 0:
        num_workers = 1

    steps_per_worker = np.full(num_workers, num_steps // num_workers)
    steps_per_worker[:num_steps % num_workers] += 1

    worker_args = [(i, steps_per_worker[i], env_name, device, max_episode_steps) for i in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        filepaths = pool.starmap(collect_sequences_worker, worker_args)

    all_data = []
    for path in filepaths:
        if path and Path(path).exists():
            try:
                all_data.append(torch.load(path))
                os.remove(path)
            except Exception as e:
                print(f"Error loading or removing temp file {path}: {e}")

    if not all_data:
        return None

    # Concatenate data from all workers
    final_buffer = {key: torch.cat([d[key] for d in all_data], dim=0) for key in all_data[0]}
    print(f"Finished collecting. Total transitions: {len(final_buffer['actions'])}.")
    return final_buffer
