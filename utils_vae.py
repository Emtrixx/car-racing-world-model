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

from utils import LatentStateWrapperVQ, preprocess_observation

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
def collect_and_save_frames(num_frames, save_dir="data/frames"):
    """
    Collects frames from the CarRacing environment using a pre-trained agent
    and saves them individually to disk.

    Args:
        num_frames (int): The total number of frames to collect and save.
        save_dir (str): The directory where frames will be saved.
    """
    print(f"Collecting and saving {num_frames} frames to '{save_dir}'...")

    # --- Create Save Directory ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # --- Load Model ---
    model_id = "Pyro-X2/CarRacingSB3"
    model_filename = "ppo-CarRacing-v3.zip"
    try:
        # Load the model from Hugging Face Hub
        checkpoint = load_from_hub(model_id, model_filename)
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        exit(1)

    # --- Environment and Model Setup ---
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    model = PPO.load(checkpoint, env=env)

    observation, info = env.reset()
    frames_collected = 0

    # --- Initial Frame Skip ---
    # Skip the first 50 frames to leave out the initial zooming-in animation.
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

        # Preprocess and save the frame
        processed_frame = preprocess_observation(observation)
        # Convert to tensor and change format from HxWxC to CxHxW
        frame_tensor = torch.tensor(processed_frame, dtype=torch.uint8).permute(2, 0, 1)

        # Save the individual frame tensor to disk
        save_path = os.path.join(save_dir, f"frame_{frames_collected:06d}.pt")
        torch.save(frame_tensor, save_path)

        frames_collected += 1
        pbar.update(1)

        if terminated or truncated:
            observation, info = env.reset()
            # Skip zoom-in on reset
            for _ in range(50):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    observation, _ = env.reset()

    pbar.close()
    env.close()
    print(f"Finished collecting and saving {frames_collected} frames in '{save_dir}'.")


# --- Data Loading ---
def load_frames(load_dir="data/frames", num_frames_to_load=None, normalize=False):
    """
    Loads frames from disk that were saved by collect_and_save_frames.

    Args:
        load_dir (str): The directory where frames are saved.
        num_frames_to_load (int, optional): The maximum number of frames to load.
                                            If None, loads all frames in the directory.
        normalize (bool): If True, converts frames to float and normalizes to [0, 1].

    Returns:
        torch.Tensor: A tensor containing all the loaded frames, stacked together.
    """
    if not os.path.isdir(load_dir):
        print(f"Error: Directory not found at {load_dir}")
        return torch.empty(0)

    print(f"Loading frames from '{load_dir}'...")
    # Get a sorted list of all frame files
    frame_files = sorted([f for f in os.listdir(load_dir) if f.endswith('.pt')])

    if not frame_files:
        print("No '.pt' files found in the directory.")
        return torch.empty(0)

    # Determine how many frames to load
    if num_frames_to_load is not None:
        frame_files = frame_files[:num_frames_to_load]

    loaded_frames = []
    # Use tqdm for a progress bar
    for filename in tqdm(frame_files, desc="Loading frames"):
        file_path = os.path.join(load_dir, filename)
        try:
            frame_tensor = torch.load(file_path)
            loaded_frames.append(frame_tensor)
        except Exception as e:
            print(f"Could not load file {filename}: {e}")

    if not loaded_frames:
        return torch.empty(0)

    # Stack all loaded frames into a single tensor
    stacked_frames = torch.stack(loaded_frames)
    stacked_frames = stacked_frames.float()

    if normalize:
        # normalize pixel values to the [0, 1] range
        stacked_frames = stacked_frames / 255.0

    print(f"Finished loading {len(loaded_frames)} frames.")
    return stacked_frames


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


# Helper function to get the LatentStateWrapperVQ instance
def get_vq_wrapper(env_instance):
    current_env = env_instance
    while hasattr(current_env, 'env') or hasattr(current_env, 'venv'):  # Check for 'venv' for VecEnv
        if isinstance(current_env, LatentStateWrapperVQ):
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

    if isinstance(current_env, LatentStateWrapperVQ):  # Check last env in chain
        return current_env
    print("Warning: LatentStateWrapperVQ not found in environment stack.")
    return None
