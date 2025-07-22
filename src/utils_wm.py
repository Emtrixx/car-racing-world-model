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
from src.utils import make_env_sb3, ENV_NAME, NUM_STACK, DEVICE, VQ_VAE_CHECKPOINT_FILENAME
from src.vq_conv_vae import VQVAE


class TransformerWorldModelDataCollector:
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

    def collect_steps(self, num_steps: int, display_progress=True):
        """Runs the PPO agent in the environment for a given number of steps."""
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()
        is_first_step_in_episode = True

        # We need the initial state's tokens to start the buffer correctly.
        prev_tokens = self.get_vq_indices(obs[-1])

        step_iterator = range(num_steps)
        if display_progress:
            step_iterator = tqdm(step_iterator, desc="Collecting Transformer Steps")

        for step in step_iterator:
            action, _ = self.ppo_agent.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            next_tokens = self.get_vq_indices(next_obs[-1])

            self.replay_buffer.append({
                "prev_tokens": prev_tokens.squeeze(0).to(torch.int64),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done or truncated], dtype=torch.float32),
                "next_tokens": next_tokens.squeeze(0).to(torch.int64),
                "is_first_step": torch.tensor([is_first_step_in_episode], dtype=torch.bool)
            })
            is_first_step_in_episode = False

            obs = next_obs
            prev_tokens = next_tokens

            if done or truncated:
                obs, _ = self.env.reset()
                prev_tokens = self.get_vq_indices(obs[-1])
                is_first_step_in_episode = True


def transformer_collect_sequences_worker(worker_id, num_steps_to_collect, env_name, device_str, max_episode_steps):
    """Worker function for parallel data collection."""
    try:
        print(f"[Worker {worker_id}] Starting...")
        env = make_env_sb3(env_id=env_name, frame_stack_num=NUM_STACK, max_episode_steps=max_episode_steps)
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str, env=env)
        vq_vae = VQVAE().to(device_str)
        vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str))
        vq_vae.eval()

        show_progress = (worker_id == 0)
        collector = TransformerWorldModelDataCollector(env, ppo_agent, vq_vae, device_str)
        collector.collect_steps(num_steps=num_steps_to_collect, display_progress=show_progress)
        env.close()

        print(f"[Worker {worker_id}] Finished. Collected {len(collector.replay_buffer)} transitions.")

        # Package data for returning
        data_to_save = {
            'prev_tokens': torch.stack([s['prev_tokens'] for s in collector.replay_buffer]),
            'actions': torch.stack([s['action'] for s in collector.replay_buffer]),
            'rewards': torch.stack([s['reward'] for s in collector.replay_buffer]),
            'dones': torch.stack([s['done'] for s in collector.replay_buffer]),
            'next_tokens': torch.stack([s['next_tokens'] for s in collector.replay_buffer]),
            'is_first_steps': torch.stack([s['is_first_step'] for s in collector.replay_buffer]),
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


def collect_data_for_transformer(num_steps, device, num_workers, env_name, max_episode_steps):
    """Manages parallel data collection and aggregates results."""
    print(f"Starting parallel collection with {num_workers} workers for {num_steps} total steps...")
    if num_workers <= 0:
        num_workers = 1

    steps_per_worker = np.full(num_workers, num_steps // num_workers)
    steps_per_worker[:num_steps % num_workers] += 1

    worker_args = [(i, steps_per_worker[i], env_name, device, max_episode_steps) for i in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        filepaths = pool.starmap(transformer_collect_sequences_worker, worker_args)

    all_data = []
    for path in filepaths:
        if path and Path(path).exists():
            try:
                all_data.append(torch.load(path, map_location='cpu'))
                os.remove(path)
            except Exception as e:
                print(f"Error loading or removing temp file {path}: {e}")

    if not all_data:
        return None

    # Concatenate data from all workers
    final_buffer = {key: torch.cat([d[key] for d in all_data], dim=0) for key in all_data[0]}
    print(f"Finished collecting. Total transitions: {len(final_buffer['actions'])}.")
    return final_buffer


class GruWorldModelDataCollector:
    """
    Collects training data for the World Model by running a pretrained policy.
    """

    def __init__(self, env, ppo_agent, vq_vae_model, device):
        self.env = env
        self.ppo_agent = ppo_agent
        self.vq_vae_model = vq_vae_model
        self.device = device
        # Use a simple deque as a replay buffer
        self.replay_buffer = deque(maxlen=250_000)

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

    def collect_steps(self, num_steps: int, display_progress=True):
        """
        Runs the PPO agent in the environment for a given number of steps.

        Args:
            num_steps (int): The total number of steps to collect.
        """
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()
        is_first_step_in_episode = True

        step_iterator = range(num_steps)
        if display_progress:
            step_iterator = tqdm(step_iterator, desc="Collecting GRU Steps")

        for step in step_iterator:
            # Get action from the pretrained PPO agent
            action, _ = self.ppo_agent.predict(obs, deterministic=False)

            # Step the environment with the action
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Get the ground truth token indices for the next observation
            next_state_tokens = self.get_vq_indices(next_obs[-1])  # access last frame in the stack

            # Store the relevant data tuple for world model training
            # We store the action, reward, done flag, and the tokenized *next* state.
            self.replay_buffer.append({
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done or truncated], dtype=torch.float32),
                "next_tokens": next_state_tokens.squeeze(0).to(torch.int64),  # Store as [16]
                "is_first_step": torch.tensor([is_first_step_in_episode], dtype=torch.bool)
            })
            is_first_step_in_episode = False

            # Update observation and reset if the episode is over
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                is_first_step_in_episode = True

        print(f"Collection complete. Final buffer size: {len(self.replay_buffer)}")


def gru_collect_sequences_worker(worker_id, num_steps_to_collect_by_worker, env_name_str,
                                 device_str_for_worker,
                                 max_episode_steps_collect_int):  # Added max_episode_steps
    try:
        import os
        import time
        import torch
        import gymnasium as gym
        from tqdm import tqdm

        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Starting, assigned {num_steps_to_collect_by_worker} steps. Device: {device_str_for_worker}")

        # --- Initialize Environment ---
        env = make_env_sb3(
            env_id=env_name_str,
            frame_stack_num=NUM_STACK,
            max_episode_steps=max_episode_steps_collect_int,
        )

        # --- Load PPO Model ---
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str_for_worker, env=env)

        # --- Load VQ-VAE Model ---
        vq_vae = VQVAE().to(device_str_for_worker)
        vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str_for_worker))
        vq_vae.eval()

        # --- Prepare for Data Collection ---
        show_progress = (worker_id == 0)  # Only show progress bar for the first worker
        collector = GruWorldModelDataCollector(env, ppo_agent, vq_vae, device_str_for_worker)
        collector.collect_steps(num_steps=num_steps_to_collect_by_worker, display_progress=show_progress)

        env.close()
        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Finished data collection. Collected {len(collector.replay_buffer)} transitions.")

        # --- Prepare Data for Saving ---
        if not collector.replay_buffer:
            print(f"[Worker {worker_id}] Replay buffer is empty. Nothing to save.")
            return None

        data_to_save = {
            'actions': torch.stack([s['action'] for s in collector.replay_buffer]),
            'rewards': torch.stack([s['reward'] for s in collector.replay_buffer]),
            'dones': torch.stack([s['done'] for s in collector.replay_buffer]),
            'next_tokens': torch.stack([s['next_tokens'] for s in collector.replay_buffer]),
            'is_first_steps': torch.stack([s['is_first_step'] for s in collector.replay_buffer]),
        }
        # Create a directory for temporary worker data if it doesn't exist
        temp_data_dir = Path("./tmp_worker_data")
        temp_data_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = temp_data_dir / f"temp_gru_worker_data_{worker_id}_{timestamp}.pt"

        try:
            torch.save(data_to_save, filename)
            return str(filename)  # Return the filepath
        except Exception as e_save:
            print(f"[Worker {worker_id}] ERROR saving data to {filename}: {e_save}")
            import traceback
            traceback.print_exc()
            return None  # Return None on save error
    except Exception as e:
        print(f"[Worker {worker_id}, PID {os.getpid()}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals() and 'env' in dir():
            env.close()
        return None  # Return None on general error in worker


def collect_sequences_for_gru(num_steps_total, device_str_main,
                              num_collection_workers_int,
                              env_name_str_for_worker,
                              max_episode_steps_collect_int
                              ):
    print(
        f"Starting parallel collection with {num_collection_workers_int} workers for {num_steps_total} total steps...")

    if num_collection_workers_int <= 0:
        print("Warning: num_collection_workers is not positive. Defaulting to 1 worker (serial collection).")
        num_collection_workers_int = 1

    steps_per_worker = np.full(num_collection_workers_int, num_steps_total // num_collection_workers_int)
    steps_per_worker[:num_steps_total % num_collection_workers_int] += 1

    worker_args_list = [
        (i, steps_per_worker[i], env_name_str_for_worker, device_str_main, max_episode_steps_collect_int)
        for i in range(num_collection_workers_int) if steps_per_worker[i] > 0
    ]

    if not worker_args_list:
        print("No steps to collect or no workers to assign. Skipping parallel collection.")
        return None

    pool_size = len(worker_args_list)
    print(f"Distributing work: {steps_per_worker} steps per worker. Spawning {pool_size} worker processes.")

    with mp.Pool(processes=pool_size) as pool:
        filepath_results = pool.starmap(gru_collect_sequences_worker, worker_args_list)

    all_worker_data_dicts = []
    for filepath_result in filepath_results:
        if filepath_result and Path(filepath_result).exists():
            try:
                worker_data = torch.load(filepath_result)
                all_worker_data_dicts.append(worker_data)
                os.remove(filepath_result)
            except Exception as e_load:
                print(f"ERROR loading or removing data from file {filepath_result}: {e_load}")
        elif filepath_result:
            print(f"Worker returned filepath {filepath_result}, but file not found.")
        else:
            print(f"A worker failed to produce a data file.")

    if not all_worker_data_dicts:
        print("No valid data loaded from any worker.")
        return None

    final_data_buffer = {
        key: torch.cat([d[key] for d in all_worker_data_dicts], dim=0)
        for key in all_worker_data_dicts[0]
    }

    print(f"Finished collecting. Total transitions from all workers: {len(final_data_buffer['actions'])}.")
    temp_data_dir = Path("./tmp_worker_data")
    if temp_data_dir.exists() and not any(temp_data_dir.iterdir()):
        try:
            temp_data_dir.rmdir()
        except OSError as e_rmdir:
            print(f"Warning: Could not remove temporary directory {temp_data_dir}: {e_rmdir}")

    return final_data_buffer
