# train_vae.py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time

# Import from local modules
from utils import (DEVICE, ENV_NAME, IMG_SIZE, CHANNELS, LATENT_DIM,
                   VAE_CHECKPOINT_FILENAME, transform)
from models.conv_vae import ConvVAE

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 35         # Number of training epochs
BETA = 1.0          # Weight for the KL divergence term
NUM_FRAMES_COLLECT = 10000 # How many frames to collect for training dataset

# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar, beta=BETA):
    # Reconstruction Loss (Binary Cross Entropy per pixel, summed over pixels/channels, averaged over batch)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return BCE + beta * KLD, BCE / (CHANNELS * IMG_SIZE * IMG_SIZE), KLD # Return normalized BCE/KLD for logging

# --- Data Collection ---
def collect_frames(env_name, num_frames, transform_fn):
    print(f"Collecting {num_frames} frames for VAE training...")
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    frame_count = 0

    while frame_count < num_frames:
        action = env.action_space.sample() # Random actions
        state, reward, terminated, truncated, info = env.step(action)
        frame = env.render()

        if frame is not None:
            processed_frame = transform_fn(frame) # Use transform from utils
            frames.append(processed_frame)
            frame_count += 1

        if terminated or truncated:
            state, _ = env.reset()

        if frame_count % 500 == 0 and frame_count > 0:
             print(f"  Collected {frame_count}/{num_frames} frames...")

    env.close()
    print(f"Finished collecting {len(frames)} frames.")
    return torch.stack(frames)

# --- Dataset Class ---
class FrameDataset(Dataset):
    def __init__(self, frame_data):
        self.data = frame_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Training Loop ---
def train_vae_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    train_loss = 0
    bce_loss_total = 0
    kld_loss_total = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss, bce_norm, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=BETA)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # Accumulate unnormalized BCE/KLD for average calculation
        bce_loss_total += bce_norm * data.size(0) * (CHANNELS * IMG_SIZE * IMG_SIZE)
        kld_loss_total += kld * data.size(0)


        if batch_idx % 50 == 0:
             print(f'  Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                   f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.4f} '
                   f'(BCE/px: {bce_norm:.6f}, KLD: {kld:.4f})')


    avg_loss = train_loss / len(dataloader.dataset)
    avg_bce = bce_loss_total / len(dataloader.dataset)
    avg_kld = kld_loss_total / len(dataloader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (Avg BCE: {avg_bce:.4f}, Avg KLD: {avg_kld:.4f})')
    return avg_loss

# --- Visualization ---
def visualize_reconstruction(model, dataloader, device, epoch, n_samples=8):
    model.eval()
    data = next(iter(dataloader)).to(device)
    if data.size(0) > n_samples: data = data[:n_samples]

    with torch.no_grad():
        recon_batch, _, _ = model(data)

    original = data.cpu()
    reconstructed = recon_batch.cpu()

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig.suptitle(f'Epoch {epoch} - Original vs. Reconstructed', fontsize=16)
    for i in range(n_samples):
        img_orig = original[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].set_title(f'Recon {i+1}')
        axes[1, i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"checkpoints/vae_reconstruction_epoch_{epoch}.png"
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close(fig) # Close the figure to free memory

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting VAE training on device: {DEVICE}")
    print(f"Using environment: {ENV_NAME}")

    # 1. Collect Data
    frame_data = collect_frames(ENV_NAME, NUM_FRAMES_COLLECT, transform) # Use transform from utils

    # 2. Create Dataset and DataLoader
    dataset = FrameDataset(frame_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Initialize Model and Optimizer
    model = ConvVAE().to(DEVICE) # Uses constants from utils implicitly via models.py
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_vae_epoch(model, dataloader, optimizer, epoch, DEVICE)
        if epoch % 5 == 0 or epoch == EPOCHS:
             visualize_reconstruction(model, dataloader, DEVICE, epoch)
    end_time = time.time()
    print(f"\nVAE Training finished in {end_time - start_time:.2f} seconds.")

    # 5. Save the trained model
    try:
        torch.save(model.state_dict(), VAE_CHECKPOINT_FILENAME) # Use path from utils
        print(f"Model saved to {VAE_CHECKPOINT_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")