# --- Main Execution ---
import argparse
import time

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils import DEVICE, ENV_NAME, VQ_VAE_CHECKPOINT_FILENAME
from src.utils_vae import FrameDataset, visualize_reconstruction, collect_and_save_frames, load_frames_from_disk
from src.vq_conv_vae import VQVAE


def get_config(name="default"):
    configs = {
        "default": {
            "num_frames_collect": 1_000_000,  # How many frames to collect for training dataset
            "batch_size": 128,  # Batch size for training
            "learning_rate": 1e-3,  # Learning rate for optimizer
            "epochs": 50,  # Number of training epochs
        },
        # for testing
        "test": {
            "num_frames_collect": 10000,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "epochs": 5,
        }
    }
    return configs.get(name, configs["default"])


def train_vqvae_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, vq_loss, _quantized, _encoding_indices = model(data)
        # Calculate the loss
        loss = vq_loss + F.mse_loss(recon_batch, data)  # VQ loss + reconstruction loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'  Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.4f}')

    avg_loss = train_loss / len(dataloader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss


if __name__ == "__main__":
    print(f"Starting VAE training on device: {DEVICE}")
    print(f"Using environment: {ENV_NAME}")

    parser = argparse.ArgumentParser(description="Train a VQ-VAE on collected frames from the environment.")
    parser.add_argument("--config", type=str, default="default", help="Configuration name to use (default/test)")
    parser.add_argument("--collect", action="store_true",
                        )
    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Collect Data if needed
    if args.collect:
        # This will collect frames and save them to the default directory
        print(f"Collecting {config['num_frames_collect']} frames from the environment...")
        collect_and_save_frames(config["num_frames_collect"])

    # Load frames from disk
    frame_data = load_frames_from_disk(max_frames_to_load=config["num_frames_collect"])

    # Create Dataset and DataLoader
    dataset = FrameDataset(frame_data)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    # Initialize Model and Optimizer
    model = VQVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_vqvae_epoch(model, dataloader, optimizer, epoch, DEVICE)
        if epoch % 5 == 0 or epoch == config["epochs"]:
            visualize_reconstruction(model, dataloader, DEVICE, epoch)
    end_time = time.time()
    print(f"\nVAE Training finished in {end_time - start_time:.2f} seconds.")

    # Save the trained model
    try:
        torch.save(model.state_dict(), VQ_VAE_CHECKPOINT_FILENAME)  # Use path from utils
        print(f"Model saved to {VQ_VAE_CHECKPOINT_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")
