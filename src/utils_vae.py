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

from src.utils import preprocess_observation, IMAGES_DIR, DATA_DIR


# --- Dataset Class ---
class FrameDataset(Dataset):
    def __init__(self, frame_data):
        self.data = frame_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collect_and_save_frames(num_frames, save_dir="data/frames", batch_size=1000, frame_skip=4):
    """
    Collects frames using a pre-trained agent and saves them to disk in batches.

    Args:
        num_frames (int): The total number of frames to collect.
        save_dir (str): The directory where frame batches will be saved.
        batch_size (int): The number of frames to save in each file.
        frame_skip (int): How many simulation steps to skip between saved frames.
    """
    print(f"Collecting {num_frames} frames, saving to '{save_dir}'...")
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    model_id = "Pyro-X2/CarRacingSB3"
    model_filename = "ppo-CarRacing-v3.zip"
    try:
        checkpoint = load_from_hub(model_id, model_filename)
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        return

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    model = PPO.load(checkpoint, env=env)

    batch_frames = []
    batch_num = 0

    observation, _ = env.reset()
    for _ in range(50):
        action, _ = model.predict(observation, deterministic=False)
        observation, _, _, _, _ = env.step(action)

    step = 0
    # Initialize tqdm progress bar
    with tqdm(total=num_frames, desc="Collecting frames") as pbar:
        while pbar.n < num_frames:
            action, _ = model.predict(observation, deterministic=False)
            observation, _, terminated, truncated, _ = env.step(action)

            if step % frame_skip == 0:
                processed_frame = preprocess_observation(observation)
                frame_tensor = torch.tensor(processed_frame, dtype=torch.float32).permute(2, 0, 1)
                batch_frames.append(frame_tensor)
                pbar.update(1)  # Increment the progress bar

                if len(batch_frames) == batch_size:
                    batch_path = os.path.join(save_dir, f"batch_{batch_num}.pt")
                    torch.save(torch.stack(batch_frames), batch_path)
                    batch_frames.clear()
                    batch_num += 1

            if terminated or truncated:
                observation, _ = env.reset()
                for _ in range(50):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, _, _, _ = env.step(action)

            step += 1

    if batch_frames:
        batch_path = os.path.join(save_dir, f"batch_{batch_num}.pt")
        torch.save(torch.stack(batch_frames), batch_path)

    env.close()
    print(f"\n✅ Finished collecting and saving {num_frames} frames.")


# --- Load All Frame Batches from Disk ---
def load_frames_from_disk(load_dir=DATA_DIR / "frames", max_frames_to_load=None):
    """
    Loads frame batches from a directory with a progress bar and an optional frame limit.

    Args:
        load_dir (str): The directory containing the saved frame batches.
        max_frames_to_load (int, optional): The maximum number of frames to load.
                                            If None, all frames are loaded. Defaults to None.

    Returns:
        torch.Tensor: A single tensor containing all loaded frames, or None if no files found.
    """
    print(f"Loading frames from '{load_dir}'...")
    frame_files = sorted(pathlib.Path(load_dir).glob('batch_*.pt'), key=lambda p: int(p.stem.split('_')[1]))

    if not frame_files:
        print(f"❌ Error: No 'batch_*.pt' files found in '{load_dir}'.")
        return None

    all_batches = []
    total_loaded = 0

    # Wrap file iteration with tqdm for a progress bar
    for file_path in tqdm(frame_files, desc="Loading batches"):
        if max_frames_to_load is not None and total_loaded >= max_frames_to_load:
            print(f"\nReached frame limit of {max_frames_to_load}. Stopping.")
            break

        batch = torch.load(file_path)

        # If loading this batch would exceed the limit, slice it
        if max_frames_to_load is not None:
            remaining_needed = max_frames_to_load - total_loaded
            if len(batch) > remaining_needed:
                batch = batch[:remaining_needed]

        all_batches.append(batch)
        total_loaded += len(batch)

    if not all_batches:
        print("No frames were loaded.")
        return None

    full_dataset = torch.cat(all_batches, dim=0)
    print(f"\n✅ Successfully loaded a total of {len(full_dataset)} frames.")
    return full_dataset


# --- Debugging Function to Display a Frame ---
def visualize_single_frame(frame_tensor, title="Sample Frame"):
    """
    Displays a single frame tensor using matplotlib.

    Args:
        frame_tensor (torch.Tensor): A single frame tensor with shape (C, H, W).
        title (str): The title for the plot.
    """
    if not isinstance(frame_tensor, torch.Tensor) or frame_tensor.dim() != 3:
        print("Error: Input must be a 3D PyTorch tensor of shape (C, H, W).")
        return

    # Create a single figure with a specific size
    plt.figure(figsize=(4, 4))

    if frame_tensor.shape[0] == 1:
        # Grayscale: remove channel dim (1, H, W) -> (H, W) and convert to numpy
        img_np = frame_tensor.squeeze().cpu().numpy()
        plt.imshow(img_np, cmap='gray')
    else:
        # RGB: convert to numpy and permute to (H, W, C)
        img_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)

    # Add title and turn off axes for the current figure
    plt.title(title)
    plt.axis('off')
    plt.show()


# --- Visualization ---
def visualize_reconstruction(model, dataloader, device, epoch, n_samples=8):
    model.eval()
    data = next(iter(dataloader)).to(device)
    if data.size(0) > n_samples:
        data = data[:n_samples]

    with torch.no_grad():
        x_recon, _loss, _quantized, _encoding_indices, _z, _perplexity = model(data)

    original = data.cpu()
    reconstructed = x_recon.cpu().clamp(0, 1)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig.suptitle(f'Epoch {epoch} - Original vs. Reconstructed', fontsize=16)
    for i in range(n_samples):
        # check if images are grayscale and display accordingly
        if original[i].dim() == 3 and original[i].shape[0] == 1:
            axes[0, i].imshow(original[i].squeeze(), cmap='gray')
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        else:
            axes[0, i].imshow(original[i].permute(1, 2, 0).numpy())
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())

        axes[0, i].set_title(f'Original {i + 1}')
        axes[1, i].set_title(f'Recon {i + 1}')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Create images directory if it doesn't exist
    save_path = IMAGES_DIR / f"vqvae_reconstruction_epoch_{epoch}.png"
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    loaded_frames = load_frames_from_disk()
    if loaded_frames is not None:
        print(f"\nShape of the loaded dataset: {loaded_frames.shape}")

        # --- Step 3: Use the debugging function to view a sample frame ---
        print("Displaying a sample frame for verification...")
        random_frame_index = np.random.randint(0, len(loaded_frames))
        # Display the 100th frame from the loaded dataset
        visualize_single_frame(loaded_frames[random_frame_index], title=f"Sample Frame {random_frame_index + 1}")
