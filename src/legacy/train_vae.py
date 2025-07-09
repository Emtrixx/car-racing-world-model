# train_vae.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Import from local modules
from src.utils import (DEVICE, ENV_NAME, IMG_SIZE, CHANNELS)
from src.legacy.utils_legacy import transform, VAE_CHECKPOINT_FILENAME
from src.legacy.conv_vae import ConvVAE
from src.utils_vae import collect_frames, FrameDataset, visualize_reconstruction

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50  # Number of training epochs
BETA = 1.0  # Weight for the KL divergence term
NUM_FRAMES_COLLECT = 50000  # How many frames to collect for training dataset


# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar, beta=BETA):
    # Reconstruction Loss (Binary Cross Entropy per pixel, summed over pixels/channels, averaged over batch)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return BCE + beta * KLD, BCE / (CHANNELS * IMG_SIZE * IMG_SIZE), KLD  # Return normalized BCE/KLD for logging


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


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting VAE training on device: {DEVICE}")
    print(f"Using environment: {ENV_NAME}")

    # 1. Collect Data
    frame_data = collect_frames(ENV_NAME, NUM_FRAMES_COLLECT, transform)  # Use transform from utils

    # 2. Create Dataset and DataLoader
    dataset = FrameDataset(frame_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Initialize Model and Optimizer
    model = ConvVAE().to(DEVICE)  # Uses constants from utils implicitly via models.py
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
        torch.save(model.state_dict(), VAE_CHECKPOINT_FILENAME)  # Use path from utils
        print(f"Model saved to {VAE_CHECKPOINT_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")
