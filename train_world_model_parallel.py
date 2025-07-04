import argparse
import multiprocessing as mp
import os
import time

import torch
import torch.nn as nn
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset

from utils import (
    ENV_NAME,  # Default: "CarRacing-v3"
    ACTION_DIM,  # Default: 3
    DEVICE, WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, WorldModelDataCollector, WorldModelTrainer
)
from vq_conv_vae import NUM_EMBEDDINGS, EMBEDDING_DIM, VQVAE
from world_model import WorldModelGRU, GRU_HIDDEN_DIM

# --- Configuration ---
# Training Hyperparameters
NUM_STEPS = 1_000_000  # Number of steps to collect for training the world model
WM_EPOCHS = 10  # Number of epochs to train the world model
WM_BATCH_SIZE = 32  # Sequences per batch
WM_LEARNING_RATE = 1e-4  # Learning rate for world model optimizer
SEQUENCE_LENGTH = 32  # Length of sequences to train on
MAX_GRAD_NORM = 1.0  # Max gradient norm for clipping

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4  # For multiprocessing data collection
NUM_LOADER_WORKERS = 4  # For DataLoader for PyTorch training

# Environment settings for data collection
MAX_EPISODE_STEPS_COLLECT = 1000  # Max steps per episode in the collection environment


def get_config(name="default"):
    configs = {
        "default": {
            "env_name": ENV_NAME,
            'action_dim': ACTION_DIM,
            'num_steps': NUM_STEPS,
            'epochs': WM_EPOCHS,
            'learning_rate': WM_LEARNING_RATE,
            'batch_size': WM_BATCH_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'max_grad_norm': MAX_GRAD_NORM,
            'max_episode_steps_collect': MAX_EPISODE_STEPS_COLLECT,
            'device': DEVICE,
            'num_collection_workers': NUM_COLLECTION_WORKERS,
            'num_loader_workers': NUM_LOADER_WORKERS,
        }
    }
    # test configuration for quick runs
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "num_steps": 1000,
        "epochs": 3,
        "batch_size": 4,
        "sequence_length": 20,
        "num_collection_workers": 2,
        "num_loader_workers": 2,
        "max_episode_steps_collect": 100,
    })
    return configs[name]


# --- Worker function for parallel data collection ---
def collect_sequences_worker(worker_id, num_steps_to_collect_by_worker, env_name_str,
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
        env = gym.make("CarRacing-v3", render_mode="rgb_array",
                       max_episode_steps=max_episode_steps_collect_int)

        # print observation and action spaces
        print(f"[Worker {worker_id}] Observation space: {env.observation_space}")
        print(f"[Worker {worker_id}] Action space: {env.action_space}")

        # --- Load PPO Model ---
        model_id = "Pyro-X2/CarRacingSB3"
        model_filename = "ppo-CarRacing-v3.zip"
        try:
            checkpoint = load_from_hub(model_id, model_filename)
        except Exception as e:
            print(f"Failed to load model from Hugging Face Hub: {e}")
            return

        ppo_agent = PPO.load(checkpoint, env=env)

        # --- Load VQ-VAE Model ---
        vq_vae = VQVAE().to(DEVICE)

        try:
            print(f"Loading trained model from: {VQ_VAE_CHECKPOINT_FILENAME}")
            vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        except FileNotFoundError:
            print("VQ-VAE Model file not found")
            return

        # --- Prepare for Data Collection ---
        collector = WorldModelDataCollector(env, ppo_agent, vq_vae, device_str_for_worker)
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
                temp_data_dir = "."

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


# --- Data Collection for GRU (Sequence Data) ---
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
        filepath_results = pool.starmap(collect_sequences_worker, worker_args_list)

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


class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for handling the structured dictionary of sequence data.
    This dataset returns entire sequences of a fixed length.
    """

    def __init__(self, data_dict, sequence_length):
        """
        Args:
            data_dict (dict): A dictionary where keys are 'actions', 'rewards', etc.,
                              and values are tensors of the entire dataset.
            sequence_length (int): The length of the sequences to return.
        """
        self.data = data_dict
        self.sequence_length = sequence_length
        # The number of possible start points for a sequence
        self.num_sequences = len(data_dict['actions']) - sequence_length + 1

    def __len__(self):
        """Returns the total number of possible sequences."""
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns a dictionary containing one sequence of data starting at the given index.
        """
        # Define the start and end of the sequence slice
        start = idx
        end = idx + self.sequence_length

        # Slice each tensor to get the data for the full sequence
        return {
            'actions': self.data['actions'][start:end],
            'rewards': self.data['rewards'][start:end],
            'dones': self.data['dones'][start:end],
            'next_tokens': self.data['next_tokens'][start:end]
        }


# --- Main Execution ---
if __name__ == "__main__":
    # Argument Parsing: Allow selecting config from command line
    parser = argparse.ArgumentParser(description="Train GRU World Model")
    parser.add_argument("--config", type=str, default="default",
                        help="Name of the configuration to use (e.g., 'default', 'test').")
    args = parser.parse_args()

    # Load the chosen configuration
    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")

    # Set multiprocessing start method - crucial for CUDA.
    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        print(f"Could not set start method (possibly already set or not allowed): {e}")
        pass

    print(f"Starting GRU World Model training on device: {config['device']}")

    # checks for checkpoint files
    if not os.path.exists(VQ_VAE_CHECKPOINT_FILENAME):
        print(
            f"CRITICAL ERROR: VAE Checkpoint {VQ_VAE_CHECKPOINT_FILENAME} not found. Exiting before starting workers.")
        exit()

    # Collect Sequence Data (Parallelized)
    print(f"Number of collection workers configured: {config['num_collection_workers']}")
    start_collect_time = time.time()
    sequence_data_buffer = collect_sequences_for_gru(
        num_steps_total=config["num_steps"],
        device_str_main=config["device"],
        num_collection_workers_int=config["num_collection_workers"],
        env_name_str_for_worker=config["env_name"],
        max_episode_steps_collect_int=config["max_episode_steps_collect"]
    )
    print(f"Sequence data collection (parallel/serial) took {time.time() - start_collect_time:.2f} seconds.")

    if not sequence_data_buffer:
        print("ERROR: No sequence data collected. Exiting.")
        exit()

    # Prepare DataLoader
    sequence_dataset = SequenceDataset(sequence_data_buffer, config["sequence_length"])
    wm_dataloader = DataLoader(
        sequence_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_loader_workers"]
    )

    # Initialize GRU World Model
    world_model_gru = WorldModelGRU(
        latent_dim=EMBEDDING_DIM,
        codebook_size=NUM_EMBEDDINGS,
        action_dim=ACTION_DIM,
        hidden_dim=GRU_HIDDEN_DIM,
    )
    world_model_gru.to(config['device'])

    # Initialize VQ-VAE Model
    vq_vae_model = VQVAE(embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS)
    vq_vae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME))
    vq_vae_model.to(config['device'])
    vq_vae_model.eval()

    # Handle DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using nn.DataParallel for GRU model training across {torch.cuda.device_count()} GPUs.")
        world_model_gru = nn.DataParallel(world_model_gru)

    # Create the trainer instance
    # The trainer encapsulates the model, optimizer, and training logic.
    trainer = WorldModelTrainer(world_model_gru, vq_vae_model, config)

    # Run the training loop
    print("Starting GRU World Model training via WorldModelTrainer...")
    start_train_time = time.time()
    # trainer saves checkpoints automatically during training
    trainer.train(wm_dataloader, num_epochs=config["epochs"])
    print(f"GRU World Model training took {time.time() - start_train_time:.2f} seconds.")

    # Save the final GRU World Model
    try:
        model_state_to_save = world_model_gru.module.state_dict() if isinstance(world_model_gru,
                                                                                nn.DataParallel) else world_model_gru.state_dict()
        torch.save(model_state_to_save, WM_CHECKPOINT_FILENAME_GRU)
        print(f"GRU World Model saved to {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e:
        print(f"Error saving GRU World Model: {e}")
