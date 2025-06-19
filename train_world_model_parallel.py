import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

from utils_vae import SB3_MODEL_PATH, get_vq_wrapper
from vq_conv_vae import VQVAE, NUM_EMBEDDINGS, EMBEDDING_DIM
from world_model import WorldModelGRU
from utils import (
    ENV_NAME,  # Default: "CarRacing-v3"
    ACTION_DIM,  # Default: 3
    NUM_STACK,  # Default: 4
    # transform is used by worker
    DEVICE, WM_CHECKPOINT_FILENAME_GRU, DEVICE_STR, VQ_VAE_CHECKPOINT_FILENAME, make_env_sb3
)
from utils_rl import PPO_ACTOR_SAVE_FILENAME

# --- Configuration ---
# GRU Model Hyperparameters
GRU_HIDDEN_DIM = 256
GRU_NUM_LAYERS = 3
GRU_INPUT_EMBED_DIM = 32  # Can be None, but GRU class handles it. For constant, it's an int or None.

# Training Hyperparameters
COLLECT_EPISODES = 1000  # Number of full episodes to collect for WM training
WM_EPOCHS = 10  # Number of epochs to train the world model
WM_BATCH_SIZE = 32  # Sequences per batch
SEQUENCE_LENGTH = 50  # Length of sequences to train on
WM_LEARNING_RATE = 1e-4  # Learning rate for world model optimizer

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4  # For multiprocessing data collection
NUM_LOADER_WORKERS = 4  # For DataLoader for PyTorch training

# Environment settings for data collection
MAX_EPISODE_STEPS_COLLECT = 400  # Max steps per episode in the collection environment


def get_config(name="default"):
    configs = {
        "default": {
            "env_name": ENV_NAME,
            "action_dim": ACTION_DIM,
            "num_stack": NUM_STACK,
            "vq_vae_checkpoint_filename": VQ_VAE_CHECKPOINT_FILENAME,
            "ppo_actor_save_filename": PPO_ACTOR_SAVE_FILENAME,
            "device_str": DEVICE_STR,
            "gru_hidden_dim": GRU_HIDDEN_DIM,
            "gru_num_layers": GRU_NUM_LAYERS,
            "gru_input_embed_dim": GRU_INPUT_EMBED_DIM,
            "wm_epochs": WM_EPOCHS,
            "wm_batch_size": WM_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "wm_learning_rate": WM_LEARNING_RATE,
            "num_collection_workers": NUM_COLLECTION_WORKERS,
            "num_loader_workers": NUM_LOADER_WORKERS,
            "collect_episodes": COLLECT_EPISODES,
            "max_episode_steps_collect": MAX_EPISODE_STEPS_COLLECT,
            "wm_checkpoint_filename_gru": WM_CHECKPOINT_FILENAME_GRU,
        }
    }
    # test configuration for quick runs
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "collect_episodes": 10,
        "wm_epochs": 1,
        "wm_batch_size": 4,
        "sequence_length": 10,
        "num_collection_workers": 2,
        "num_loader_workers": 2,
        "max_episode_steps_collect": 100,
    })
    return configs[name]


# --- Worker function for parallel data collection ---
def collect_sequences_worker(worker_id, num_episodes_to_collect_by_worker, env_name_str,
                             sequence_length_int, device_str_for_worker, num_stack_int,
                             max_episode_steps_collect_int):  # Added max_episode_steps
    try:
        import os
        import time
        import torch
        import gymnasium as gym

        from legacy.conv_vae import ConvVAE
        from legacy.actor_critic import Actor
        from utils_rl import PPOPolicyWrapper
        from utils import transform, preprocess_and_encode, preprocess_and_encode_stack, FrameStackWrapper

        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Starting, assigned {num_episodes_to_collect_by_worker} episodes. Device: {device_str_for_worker}")

        # Load VQ-VAE Model for this worker
        vq_vae_model_worker = VQVAE().to(device_str_for_worker)
        vq_vae_model_worker.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str_for_worker))
        vq_vae_model_worker.eval()
        # print(f"[Worker {worker_id}] VAE loaded.")

        # Initialize Environment for this worker
        try:
            worker_env = make_env_sb3(
                env_id=env_name_str,
                vq_vae_model_instance=vq_vae_model_worker,
                transform_function=transform,
                frame_stack_num=num_stack_int,
                device_for_vae=device_str_for_worker,
                gamma=0.99,  # Standard gamma, used by NormalizeReward
                render_mode="rgb_array",  # Use rgb_array for frame collection
                max_episode_steps=max_episode_steps_collect_int,
            )
            print("Environment created successfully with make_env_sb3.")
        except Exception as e:
            print(f"Error creating environment with make_env_sb3: {e}")
            import traceback
            traceback.print_exc()
            return

        # Load Policy Model for this worker
        print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
        if not SB3_MODEL_PATH.exists():
            print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
            if hasattr(worker_env, 'close'): worker_env.close()
            return
        try:
            ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str_for_worker,
                                 env=worker_env)  # Provide env for action/obs space checks
            print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
        except Exception as e:
            print(f"ERROR loading SB3 PPO agent: {e}")
            if hasattr(worker_env, 'close'): worker_env.close()
            import traceback
            traceback.print_exc()
            return
        # print(f"[Worker {worker_id}] Policy loaded.")

        # Get the VQ wrapper from the environment to access single latent state
        vq_wrapper = get_vq_wrapper(worker_env)

        worker_collected_sequences = []
        for episode_idx in range(num_episodes_to_collect_by_worker):
            obs_stack_latent, _ = worker_env.reset()

            ep_Z_inputs_cpu = []
            ep_a_actions_cpu = []
            ep_r_rewards_cpu = []
            ep_d_dones_cpu = []
            ep_z_single_targets_cpu = []

            done, truncated = False, False
            current_steps_in_episode = 0

            while not done and not truncated:
                ep_Z_inputs_cpu.append(torch.tensor(obs_stack_latent, dtype=torch.float32).cpu())
                # 2. Get action based on PPO policy's view of the state (stacked latents)
                action_from_agent, _states = ppo_agent.predict(obs_stack_latent)
                action_tensor = torch.tensor(action_from_agent, dtype=torch.float32)
                ep_a_actions_cpu.append(action_tensor.cpu())

                next_obs_latent, reward, terminated, truncated, info = worker_env.step(action_from_agent)
                current_done_flag = terminated or truncated

                ep_r_rewards_cpu.append(torch.tensor(reward, dtype=torch.float32).cpu())
                ep_d_dones_cpu.append(torch.tensor(current_done_flag, dtype=torch.float32).cpu())

                # 3. get only one latent obs from concatenated stack by splitting into NUM_STACK parts
                with torch.no_grad():
                    z_prime_tp1_gpu = vq_wrapper.last_quantized_latent_for_render
                ep_z_single_targets_cpu.append(z_prime_tp1_gpu.cpu())

                obs_stack_latent = next_obs_latent
                done = current_done_flag
                current_steps_in_episode += 1

            num_transitions_in_episode = len(ep_a_actions_cpu)
            if num_transitions_in_episode >= sequence_length_int:
                for i in range(num_transitions_in_episode - sequence_length_int + 1):
                    z_input_s = torch.stack(ep_Z_inputs_cpu[i: i + sequence_length_int])  # Use renamed list
                    a_input_s = torch.stack(ep_a_actions_cpu[i: i + sequence_length_int])
                    z_target_s = torch.stack(ep_z_single_targets_cpu[i: i + sequence_length_int])
                    r_target_s = torch.stack(ep_r_rewards_cpu[i: i + sequence_length_int]).unsqueeze(-1)
                    d_target_s = torch.stack(ep_d_dones_cpu[i: i + sequence_length_int]).unsqueeze(-1)
                    worker_collected_sequences.append((z_input_s, a_input_s, z_target_s, r_target_s, d_target_s))

            if (episode_idx + 1) % 5 == 0 or episode_idx == num_episodes_to_collect_by_worker - 1:
                print(
                    f"  [Worker {worker_id}, PID {os.getpid()}] Ep {episode_idx + 1}/{num_episodes_to_collect_by_worker}. Steps: {current_steps_in_episode}. Worker seqs: {len(worker_collected_sequences)}")

        worker_env.close()
        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Finished data collection. Collected {len(worker_collected_sequences)} sequences.")

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
            torch.save(worker_collected_sequences, filename)
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
            worker_env.close()
        return None  # Return None on general error in worker


# --- Data Collection for GRU (Sequence Data) ---
def collect_sequences_for_gru(num_episodes_total, sequence_length_int, device_str_main,
                              num_collection_workers_int,
                              env_name_str_for_worker,
                              num_stack_int_for_worker,
                              max_episode_steps_collect_int
                              ):
    print(
        f"Starting parallel collection with {num_collection_workers_int} workers for {num_episodes_total} total episodes...")

    if num_collection_workers_int <= 0:
        print("Warning: num_collection_workers is not positive. Defaulting to 1 worker (serial collection).")
        num_collection_workers_int = 1

    episodes_per_worker = [num_episodes_total // num_collection_workers_int] * num_collection_workers_int
    remainder_episodes = num_episodes_total % num_collection_workers_int
    for i in range(remainder_episodes):
        episodes_per_worker[i] += 1

    worker_args_list = []
    actual_workers_to_spawn = 0
    for worker_id in range(num_collection_workers_int):
        if episodes_per_worker[worker_id] == 0:
            print(f"Skipping worker {worker_id} as it has no episodes assigned.")
            continue
        actual_workers_to_spawn += 1
        args = (
            worker_id,
            episodes_per_worker[worker_id],
            env_name_str_for_worker,
            sequence_length_int,
            device_str_main,
            num_stack_int_for_worker,
            max_episode_steps_collect_int
        )
        worker_args_list.append(args)

    all_collected_sequences = []
    if not worker_args_list:
        print("No episodes to collect or no workers to assign after distribution. Skipping parallel collection.")
        return []

    # actual_workers_to_spawn should be used for Pool size if it can be less than num_collection_workers_int
    # due to low total episode count.
    pool_size = min(actual_workers_to_spawn, num_collection_workers_int)
    if pool_size == 0:  # Should be caught by "if not worker_args_list" but as a safeguard.
        print("No workers to spawn. Returning empty list.")
        return []

    print(f"Distributing work: {episodes_per_worker} episodes per worker. Spawning {pool_size} worker processes.")

    # Note: mp.set_start_method should be called once in if __name__ == "__main__"
    with mp.Pool(processes=pool_size) as pool:
        # Results will now be filepaths or None
        filepath_results = pool.starmap(collect_sequences_worker, worker_args_list)

    for worker_idx, filepath_result in enumerate(filepath_results):
        worker_actual_id = worker_args_list[worker_idx][0]  # Get actual worker_id from args
        if filepath_result and os.path.exists(filepath_result):
            try:
                print(f"Loading data from worker {worker_actual_id}'s file: {filepath_result}")
                worker_data = torch.load(filepath_result)
                all_collected_sequences.extend(worker_data)
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

    print(f"Finished collecting. Total sequences from all workers: {len(all_collected_sequences)}.")
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

    return all_collected_sequences


class SequenceDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


# --- GRU World Model Training Loop (with r, d loss) ---
def train_world_model_gru_epoch(world_model_gru, dataloader, optimizer,
                                z_criterion, r_criterion, d_criterion, epoch, device):
    world_model_gru.train()
    epoch_loss, epoch_z_loss, epoch_r_loss, epoch_d_loss = 0, 0, 0, 0
    processed_batches = 0

    for z_input_seq, a_input_seq, z_target_seq, r_target_seq, d_target_seq in dataloader:
        z_input_seq, a_input_seq, z_target_seq, r_target_seq, d_target_seq = \
            z_input_seq.to(device), a_input_seq.to(device), z_target_seq.to(device), \
                r_target_seq.to(device), d_target_seq.to(device)

        optimizer.zero_grad()
        z_pred_seq, r_pred_seq, d_pred_logits_seq, _ = world_model_gru(z_input_seq, a_input_seq)

        loss_z = z_criterion(z_pred_seq, z_target_seq)
        loss_r = r_criterion(r_pred_seq, r_target_seq)
        # Use BCEWithLogitsLoss for done flags as they are binary
        loss_d = d_criterion(d_pred_logits_seq, d_target_seq)  # Target should be float 0.0 or 1.0

        # Combine losses (can weight them)
        total_loss = loss_z + loss_r + loss_d

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_z_loss += loss_z.item()
        epoch_r_loss += loss_r.item()
        epoch_d_loss += loss_d.item()
        processed_batches += 1

    avg_loss = epoch_loss / processed_batches
    avg_z_loss = epoch_z_loss / processed_batches
    avg_r_loss = epoch_r_loss / processed_batches
    avg_d_loss = epoch_d_loss / processed_batches
    print(f'====> GRU WM Epoch: {epoch} Avg Loss: {avg_loss:.4f} '
          f'(Z: {avg_z_loss:.4f}, R: {avg_r_loss:.4f}, D: {avg_d_loss:.4f})')
    return avg_loss


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Argument Parsing: Allow selecting config from command line
    parser = argparse.ArgumentParser(description="Train GRU World Model")
    parser.add_argument("--config", type=str, default="default",
                        help="Name of the configuration to use (e.g., 'default', 'test').")
    args = parser.parse_args()

    # Load the chosen configuration
    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")

    # Use a single `config` object for all parameters
    env_name = config["env_name"]
    # This replaces all the individual constant imports/definitions at the top
    action_dim = config["action_dim"]
    num_stack = config["num_stack"]
    device_str = config["device_str"]
    device = torch.device(device_str)  # Re-initialize DEVICE based on config string

    gru_hidden_dim = config["gru_hidden_dim"]
    gru_num_layers = config["gru_num_layers"]
    gru_input_embed_dim = config["gru_input_embed_dim"]

    collect_episodes = config["collect_episodes"]
    wm_epochs = config["wm_epochs"]
    wm_batch_size = config["wm_batch_size"]
    sequence_length = config["sequence_length"]
    wm_learning_rate = config["wm_learning_rate"]

    num_collection_workers = config["num_collection_workers"]
    num_loader_workers = config["num_loader_workers"]
    max_episode_steps_collect = config["max_episode_steps_collect"]

    # Set multiprocessing start method - crucial for CUDA.
    try:
        mp.set_start_method('spawn')
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method (possibly already set or not allowed): {e}")
        pass

    print(f"Starting GRU World Model training on device: {device}")

    # VAE and Policy loading in main process are skipped if using parallel collection,
    # as workers handle their own loading. Add checks for checkpoint files.
    if not os.path.exists(VQ_VAE_CHECKPOINT_FILENAME):
        print(
            f"CRITICAL ERROR: VAE Checkpoint {VQ_VAE_CHECKPOINT_FILENAME} not found. Exiting before starting workers.")
        exit()
    if not os.path.exists(SB3_MODEL_PATH):
        print(
            f"CRITICAL ERROR: Policy Checkpoint {SB3_MODEL_PATH} not found. Exiting before starting workers.")
        exit()
    print(f"Found VAE checkpoint: {VQ_VAE_CHECKPOINT_FILENAME}")
    print(f"Found Policy checkpoint: {SB3_MODEL_PATH}")

    # Collect Sequence Data (Parallelized)
    print(f"Number of collection workers configured: {num_collection_workers}")
    start_collect_time = time.time()

    sequence_data_buffer = collect_sequences_for_gru(
        num_episodes_total=collect_episodes,
        sequence_length_int=sequence_length,
        device_str_main=device_str,
        num_collection_workers_int=num_collection_workers,
        env_name_str_for_worker=env_name,
        num_stack_int_for_worker=num_stack,
        max_episode_steps_collect_int=max_episode_steps_collect  # Pass this from config
    )

    print(f"Sequence data collection (parallel/serial) took {time.time() - start_collect_time:.2f} seconds.")

    if not sequence_data_buffer:
        print("ERROR: No sequence data collected. Exiting.")
        exit()

    # 5. Prepare DataLoader
    sequence_dataset = SequenceDataset(sequence_data_buffer)
    wm_dataloader = DataLoader(sequence_dataset, batch_size=wm_batch_size, shuffle=True, num_workers=num_loader_workers)

    # 6. Initialize GRU World Model, Optimizer, Criterion
    world_model_gru = WorldModelGRU(
        codebook_size=NUM_EMBEDDINGS,
        token_embedding_dim=EMBEDDING_DIM,
        action_dim=action_dim,
        gru_hidden_dim=gru_hidden_dim,
        gru_num_layers=gru_num_layers,
        gru_input_embed_dim=gru_input_embed_dim
    )

    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"CUDA available. Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 1:
            print("Using nn.DataParallel for GRU model training.")
            world_model_gru = nn.DataParallel(world_model_gru)

    world_model_gru.to(device)
    print(f"GRU World Model moved to device: {device}")

    wm_optimizer = optim.Adam(world_model_gru.parameters(), lr=wm_learning_rate)
    z_loss_criterion = nn.MSELoss()
    r_loss_criterion = nn.MSELoss()
    d_loss_criterion = nn.BCEWithLogitsLoss()

    # 7. Train the GRU World Model
    print("Starting GRU World Model training...")
    start_train_time = time.time()
    wm_losses = []
    for epoch in range(1, WM_EPOCHS + 1):
        loss = train_world_model_gru_epoch(world_model_gru, wm_dataloader, wm_optimizer,
                                           z_loss_criterion, r_loss_criterion, d_loss_criterion,
                                           epoch, DEVICE)
        wm_losses.append(loss)
    print(f"GRU World Model training took {time.time() - start_train_time:.2f} seconds.")

    # 8. Plot and Save Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, WM_EPOCHS + 1), wm_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"GRU World Model Training Loss (SeqLen {sequence_length})")
    plt.grid(True)
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    loss_plot_path = f"{images_dir}/{env_name}_worldmodel_gru_loss_seq{sequence_length}_e{wm_epochs}.png"
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")
    plt.close()

    # 9. Save the trained GRU World Model
    try:
        model_state_to_save = world_model_gru.module.state_dict() if isinstance(world_model_gru,
                                                                                nn.DataParallel) else world_model_gru.state_dict()
        torch.save(model_state_to_save, WM_CHECKPOINT_FILENAME_GRU)
        print(f"GRU World Model saved to {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e:
        print(f"Error saving GRU World Model: {e}")
