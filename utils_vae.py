import pathlib

import gymnasium as gym
import numpy as np
import torch
from huggingface_sb3 import load_from_hub
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch.utils.data import Dataset

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


# --- Data Collection --- (uses pre-trained SB3 PPO agent)
def collect_frames(num_frames):
    print(f"Collecting {num_frames} frames for VAE training...")

    model_id = "Pyro-X2/CarRacingSB3"
    model_filename = "ppo-CarRacing-v3.zip"

    try:
        checkpoint = load_from_hub(model_id, model_filename)  # Load the model from Hugging Face Hub
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        exit(1)

    # Create the environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    # Load the model
    model = PPO.load(checkpoint, env=env)

    frames = []
    observation, info = env.reset()

    # Skip the first 50 frames to leave out initial zooming in
    for _ in range(50):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

    # Collect frames
    frame_skip = 4  # Skip frames to reduce data size
    frame_count = 0
    for i in range(num_frames * frame_skip):
        action, _state = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        if i % frame_skip == 0:  # Collect every 4th frame

            if observation is not None:
                processed_frame = preprocess_observation(observation)
                processed_frame = torch.tensor(processed_frame, dtype=torch.float32).permute(2, 0,
                                                                                             1)  # Convert to CxHxW format
                frames.append(processed_frame)
                frame_count += 1

            if frame_count % 500 == 0 and frame_count > 0:
                print(f"  Collected {frame_count}/{num_frames} frames...")
        if terminated or truncated:
            observation, info = env.reset()
            # Skip the first 50 frames to leave out initial zooming in
            for _ in range(50):
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)

    env.close()
    print(f"Finished collecting {len(frames)} frames.")
    return torch.stack(frames)


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
