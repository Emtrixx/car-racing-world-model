import os
import pathlib

import gymnasium as gym
import numpy as np
import torch
from huggingface_sb3 import load_from_hub
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import preprocess_observation, VaeEncodeWrapper

# SB3_MODEL_FILENAME = f"sb3_default_carracing-v3_best/best_model.zip"  # best
SB3_MODEL_FILENAME = f"sb3_default_carracing-v3/ppo_model_4249320_steps.zip"  # one
SB3_MODEL_PATH = pathlib.Path("checkpoints") / SB3_MODEL_FILENAME


# --- Dataset Class ---
class FrameDataset(Dataset):
    def __init__(self, frame_data):
        self.data = frame_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Data Collection and Saving ---
def collect_and_save_frames(num_frames, save_dir="data/frames", batch_size=1000):
    """
    Collects frames and saves them to disk in batches to avoid creating too many files.

    Args:
        num_frames (int): The total number of frames to collect and save.
        save_dir (str): The directory where frame batches will be saved.
        batch_size (int): The number of frames to save in each file.
    """
    print(f"Collecting and saving {num_frames} frames to '{save_dir}' in batches of {batch_size}...")

    # --- Create Save Directory ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # --- Load Model ---
    model_id = "Pyro-X2/CarRacingSB3"
    model_filename = "ppo-CarRacing-v3.zip"
    try:
        checkpoint = load_from_hub(model_id, model_filename)
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        exit(1)

    # --- Environment and Model Setup ---
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    model = PPO.load(checkpoint, env=env)

    observation, info = env.reset()
    frames_collected = 0
    batch_counter = 0
    current_batch = []

    # --- Initial Frame Skip ---
    for _ in range(50):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    # --- Frame Collection Loop ---
    print("Starting frame collection loop...")
    pbar = tqdm(total=num_frames, desc="Collecting frames")
    while frames_collected < num_frames:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)

        processed_frame = preprocess_observation(observation)
        frame_tensor = torch.tensor(processed_frame, dtype=torch.uint8).permute(2, 0, 1)
        current_batch.append(frame_tensor)

        # If batch is full, save it to disk
        if len(current_batch) == batch_size:
            batch_tensor = torch.stack(current_batch)
            save_path = os.path.join(save_dir, f"batch_{batch_counter:04d}.pt")
            torch.save(batch_tensor, save_path)
            batch_counter += 1
            current_batch = []  # Reset for the next batch

        frames_collected += 1
        pbar.update(1)

        if terminated or truncated:
            observation, info = env.reset()
            for _ in range(50):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    observation, _ = env.reset()

    # Save any remaining frames in the last batch
    if current_batch:
        batch_tensor = torch.stack(current_batch)
        save_path = os.path.join(save_dir, f"batch_{batch_counter:04d}.pt")
        torch.save(batch_tensor, save_path)

    pbar.close()
    env.close()
    print(f"Finished collecting. Saved {frames_collected} frames in {batch_counter + 1} batches.")


# --- Data Loading (Batched) ---
def load_frames(load_dir="data/frames", num_frames_to_load=None, normalize=True):
    """
    Loads frames from batched files.

    Args:
        load_dir (str): The directory where frame batches are saved.
        num_frames_to_load (int, optional): The maximum number of frames to load.
                                            If None, loads all frames.
        normalize (bool): If True, normalizes pixel values to the [0, 1] range.

    Returns:
        torch.Tensor: A tensor containing all the loaded frames.
    """
    if not os.path.isdir(load_dir):
        print(f"Error: Directory not found at {load_dir}")
        return torch.empty(0)

    print(f"Loading batched frames from '{load_dir}'...")
    batch_files = sorted([f for f in os.listdir(load_dir) if f.startswith('batch_') and f.endswith('.pt')])

    if not batch_files:
        print("No 'batch_*.pt' files found in the directory.")
        return torch.empty(0)

    loaded_batches = []
    for filename in tqdm(batch_files, desc="Loading batches"):
        file_path = os.path.join(load_dir, filename)
        try:
            batch_tensor = torch.load(file_path)
            loaded_batches.append(batch_tensor)
        except Exception as e:
            print(f"Could not load batch file {filename}: {e}")

    if not loaded_batches:
        return torch.empty(0)

    # Concatenate all batches into a single tensor
    all_frames = torch.cat(loaded_batches, dim=0)

    # If a specific number of frames is requested, slice the tensor
    if num_frames_to_load is not None:
        all_frames = all_frames[:num_frames_to_load]

    # Convert to float, as models expect float tensors.
    all_frames = all_frames.float()

    if normalize:
        # Normalize pixel values to the [0, 1] range
        all_frames = all_frames / 255.0

    print(f"Finished loading {all_frames.shape[0]} frames.")
    return all_frames


# --- Visualization ---
def visualize_reconstruction(model, dataloader, device, epoch, n_samples=8):
    model.eval()
    data = next(iter(dataloader)).to(device)
    if data.size(0) > n_samples: data = data[:n_samples]

    with torch.no_grad():
        recon_batch, _, _, _ = model(data)

    original = data.cpu()
    reconstructed = recon_batch.cpu()

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig.suptitle(f'Epoch {epoch} - Original vs. Reconstructed', fontsize=16)
    for i in range(n_samples):
        img_orig = original[i].squeeze().numpy()
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].set_title(f'Original {i + 1}')
        axes[0, i].axis('off')
        img_recon = reconstructed[i].squeeze().numpy()
        axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].set_title(f'Recon {i + 1}')
        axes[1, i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"images/vqvae_reconstruction_epoch_{epoch}.png"
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close(fig)  # Close the figure to free memory


# Helper function to get the VaeEncodeWrapper instance
def get_vq_wrapper(env_instance):
    current_env = env_instance
    while hasattr(current_env, 'env') or hasattr(current_env, 'venv'):  # Check for 'venv' for VecEnv
        if isinstance(current_env, VaeEncodeWrapper):
            return current_env
        if hasattr(current_env, 'venv'):  # If it's a VecEnv, access its environments
            # For VecEnv, we need to get the attribute from the first environment
            # This assumes that all sub-environments are wrapped identically.
            # This part might need adjustment if using VecEnv and wanting a specific sub-env's wrapper.
            # For this script, make_env_sb3 creates a single env, so 'env' chain is more likely.
            if hasattr(current_env.venv, 'envs') and current_env.venv.envs:
                current_env = current_env.venv.envs[0]  # Check first sub-env
            else:  # Fallback or if it's not a typical VecEnv structure
                current_env = current_env.env if hasattr(current_env, 'env') else current_env.venv

        elif hasattr(current_env, 'env'):
            current_env = current_env.env
        else:
            break  # No more 'env' or 'venv' attributes

    if isinstance(current_env, VaeEncodeWrapper):  # Check last env in chain
        return current_env
    print("Warning: VaeEncodeWrapper not found in environment stack.")
    return None
