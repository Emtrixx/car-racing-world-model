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


def transformer_collect_sequences_worker(worker_id, num_steps_to_collect, env_name, device_str, max_episode_steps):
    """Worker function for parallel data collection."""
    try:
        print(f"[Worker {worker_id}] Starting...")
        env = make_env_sb3(env_id=env_name, frame_stack_num=NUM_STACK, max_episode_steps=max_episode_steps)
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str, env=env)
        vq_vae = VQVAE().to(device_str)
        vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str))
        vq_vae.eval()

        collector = TransformerWorldModelDataCollector(env, ppo_agent, vq_vae, device_str)
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
            loss, quantized_out, perplexity, encoding_indices_out = self.vq_vae_model.vq_layer(z_continuous)

        return encoding_indices_out.view(1, -1)  # Flatten to [1, 16]

    def collect_steps(self, num_steps: int):
        """
        Runs the PPO agent in the environment for a given number of steps.

        Args:
            num_steps (int): The total number of steps to collect.
        """
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()

        for step in range(num_steps):
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
                "done": torch.tensor([done], dtype=torch.float32),
                "next_tokens": next_state_tokens.squeeze(0).to(torch.int64)  # Store as [16]
            })

            # Update observation and reset if the episode is over
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                print(f"Episode finished. Buffer size: {len(self.replay_buffer)}")

        print(f"Collection complete. Final buffer size: {len(self.replay_buffer)}")


def gru_collect_sequences_worker(worker_id, num_steps_to_collect_by_worker, env_name_str,
                                 device_str_for_worker,
                                 max_episode_steps_collect_int):  # Added max_episode_steps
    try:
        import os
        import time
        import torch
        import gymnasium as gym

        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Starting, assigned {num_steps_to_collect_by_worker} episodes. Device: {device_str_for_worker}")

        # --- Initialize Environment ---
        env = env = make_env_sb3(
            env_id=ENV_NAME,
            frame_stack_num=NUM_STACK,
            gamma=0.99,  # Standard gamma, used by NormalizeReward
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps_collect_int,  # Use the max steps from worker args
        )

        # print observation and action spaces
        print(f"[Worker {worker_id}] Observation space: {env.observation_space}")
        print(f"[Worker {worker_id}] Action space: {env.action_space}")

        # --- Load PPO Model ---
        print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
        if not SB3_MODEL_PATH.exists():
            print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
            if hasattr(env, 'close'): env.close()
            return
        try:
            ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env,
                                 deterministic=False)  # deterministic=False for exploration
            print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
        except Exception as e:
            print(f"ERROR loading SB3 PPO agent: {e}")
            if hasattr(env, 'close'): env.close()
            import traceback
            traceback.print_exc()
            return

        # --- Load VQ-VAE Model ---
        vq_vae = VQVAE().to(DEVICE)

        try:
            print(f"Loading trained model from: {VQ_VAE_CHECKPOINT_FILENAME}")
            vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        except FileNotFoundError:
            print("VQ-VAE Model file not found")
            return

        # --- Prepare for Data Collection ---
        collector = GruWorldModelDataCollector(env, ppo_agent, vq_vae, device_str_for_worker)
        collector.collect_steps(num_steps=num_steps_to_collect_by_worker)

        env.close()
        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Finished data collection. Collected {len(collector.replay_buffer)} sequences.")

        # --- Prepare Data for Saving ---
        actions = torch.stack([s['action'] for s in collector.replay_buffer])
        rewards = torch.stack([s['reward'] for s in collector.replay_buffer])
        dones = torch.stack([s['done'] for s in collector.replay_buffer])
        next_tokens = torch.stack([s['next_tokens'] for s in collector.replay_buffer])

        data_to_save = {
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_tokens': next_tokens
        }
        # Create a directory for temporary worker data if it doesn't exist
        temp_data_dir = "./tmp_worker_data"
        if not os.path.exists(temp_data_dir):
            try:
                os.makedirs(temp_data_dir)
                print(f"[Worker {worker_id}] Created directory: {temp_data_dir}")
            except OSError as e:
                print(f"[Worker {worker_id}] Error creating directory {temp_data_dir}: {e}")
                # Fallback to current directory if subdir creation fails
                temp_data_dir = ".."

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = os.path.join(temp_data_dir, f"temp_worker_data_{worker_id}_{timestamp}.pt")

        try:
            print(f"[Worker {worker_id}] Attempting to save data to {filename}...")
            torch.save(data_to_save, filename)
            print(f"[Worker {worker_id}] Successfully saved data to {filename}.")
            return filename  # Return the filepath
        except Exception as e_save:
            print(f"[Worker {worker_id}] ERROR saving data to {filename}: {e_save}")
            import traceback
            traceback.print_exc()
            return None  # Return None on save error
    except Exception as e:
        print(f"[Worker {worker_id}, PID {os.getpid()}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        if 'worker_env' in locals():  # Ensure env is closed if it was initialized
            env.close()
        return None  # Return None on general error in worker


def collect_sequences_for_gru(num_steps_total, device_str_main,
                              num_collection_workers_int,
                              env_name_str_for_worker,
                              max_episode_steps_collect_int
                              ):
    print(
        f"Starting parallel collection with {num_collection_workers_int} workers for {num_steps_total} total episodes...")

    if num_collection_workers_int <= 0:
        print("Warning: num_collection_workers is not positive. Defaulting to 1 worker (serial collection).")
        num_collection_workers_int = 1

    steps_per_worker = [num_steps_total // num_collection_workers_int] * num_collection_workers_int
    remainder_episodes = num_steps_total % num_collection_workers_int
    for i in range(remainder_episodes):
        steps_per_worker[i] += 1

    worker_args_list = []
    actual_workers_to_spawn = 0
    for worker_id in range(num_collection_workers_int):
        if steps_per_worker[worker_id] == 0:
            print(f"Skipping worker {worker_id} as it has no episodes assigned.")
            continue
        actual_workers_to_spawn += 1
        args = (
            worker_id,
            steps_per_worker[worker_id],
            env_name_str_for_worker,
            device_str_main,
            max_episode_steps_collect_int
        )
        worker_args_list.append(args)

    if not worker_args_list:
        print("No episodes to collect or no workers to assign after distribution. Skipping parallel collection.")
        return []

    # actual_workers_to_spawn should be used for Pool size if it can be less than num_collection_workers_int
    # due to low total episode count.
    pool_size = min(actual_workers_to_spawn, num_collection_workers_int)
    if pool_size == 0:  # Should be caught by "if not worker_args_list" but as a safeguard.
        print("No workers to spawn. Returning empty list.")
        return []

    print(f"Distributing work: {steps_per_worker} episodes per worker. Spawning {pool_size} worker processes.")

    # Note: mp.set_start_method should be called once in if __name__ == "__main__"
    with mp.Pool(processes=pool_size) as pool:
        # Results will now be filepaths or None
        filepath_results = pool.starmap(gru_collect_sequences_worker, worker_args_list)

    all_worker_data_dicts = []
    for worker_idx, filepath_result in enumerate(filepath_results):
        worker_actual_id = worker_args_list[worker_idx][0]  # Get actual worker_id from args
        if filepath_result and os.path.exists(filepath_result):
            try:
                print(f"Loading data from worker {worker_actual_id}'s file: {filepath_result}")
                worker_data = torch.load(filepath_result, weights_only=False)
                all_worker_data_dicts.append(worker_data)
                print(
                    f"Successfully loaded {len(worker_data)} sequences from worker {worker_actual_id} (file: {filepath_result}).")
                # Optionally, delete the temporary file after successful loading
                try:
                    os.remove(filepath_result)
                    print(f"Removed temporary file: {filepath_result}")
                except OSError as e_remove:
                    print(f"Warning: Could not remove temporary file {filepath_result}: {e_remove}")
            except Exception as e_load:
                print(f"ERROR loading data from worker {worker_actual_id}'s file {filepath_result}: {e_load}")
        elif filepath_result:  # Filepath was returned but does not exist
            print(f"Worker {worker_actual_id} returned filepath {filepath_result}, but file not found.")
        else:  # Worker returned None (either general error or save error)
            print(f"Worker {worker_actual_id} failed to produce a data file.")

    if not all_worker_data_dicts:
        print("No valid data loaded from any worker.")
        return None

    final_data_buffer = {
        'actions': torch.cat([d['actions'] for d in all_worker_data_dicts], dim=0),
        'rewards': torch.cat([d['rewards'] for d in all_worker_data_dicts], dim=0),
        'dones': torch.cat([d['dones'] for d in all_worker_data_dicts], dim=0),
        'next_tokens': torch.cat([d['next_tokens'] for d in all_worker_data_dicts], dim=0),
    }

    print(f"Finished collecting. Total transitions from all workers: {len(final_data_buffer['actions'])}.")
    # Clean up the temporary directory if it's empty and was created
    temp_data_dir = "./tmp_worker_data"
    if os.path.exists(temp_data_dir) and not os.listdir(temp_data_dir):
        try:
            os.rmdir(temp_data_dir)
            print(f"Removed empty temporary directory: {temp_data_dir}")
        except OSError as e_rmdir:
            print(f"Warning: Could not remove temporary directory {temp_data_dir}: {e_rmdir}")
    elif os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
        print(f"Warning: Temporary directory {temp_data_dir} is not empty. Manual cleanup might be needed.")

    return final_data_buffer
