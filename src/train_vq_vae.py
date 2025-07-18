# --- Main Execution ---
import argparse
import time

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils import DEVICE, ENV_NAME, VQ_VAE_CHECKPOINT_FILENAME, VQ_VAE_CHECKPOINT_BEST_FILENAME
from src.utils_vae import FrameDataset, visualize_reconstruction, collect_and_save_frames, load_frames_from_disk
from src.vq_conv_vae import VQVAE, LPIPSLoss


def get_config(name="default"):
    configs = {
        "default": {
            "num_frames_collect": 1_000_000,  # How many frames to collect for training dataset
            "batch_size": 128,  # Batch size for training
            "learning_rate": 1e-3,  # Learning rate for optimizer
            "epochs": 50,  # Number of training epochs
            "num_batches_for_init": 50,  # Number of batches to use for initializing the codebook
            "perceptual_loss_weight": 0.1,  # Weight for perceptual loss in total loss calculation
            "ema_decay": 0.99,  # EMA decay parameter for VQ-VAE
            "ema_epsilon": 1e-5,  # EMA epsilon for VQ-VAE
            "validation_split": 0.1,  # train/val split
            "patience": 10,  # early stopping
            "min_delta": 0.001  # early stopping
        },
    }
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "num_frames_collect": 10000,
        "batch_size": 32,
        "epochs": 5,
        "num_batches_for_init": 10,
        "perceptual_loss_weight": 0.5,  # Weight for perceptual loss in total loss calculation
    }
    )
    return configs.get(name, configs["default"])


def train_vqvae_epoch(model, dataloader, optimizer, epoch, device, perceptual_loss, perceptual_loss_weight=0.1):
    """
    Trains the VQ-VAE model for one epoch.
    """
    model.train()

    # Trackers for metrics
    total_train_loss = 0
    total_recon_loss = 0
    total_p_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # The model returns the VQ loss, the reconstructed data, and the perplexity
        x_recon, vq_loss, quantized, _encoding_indices, z, perplexity = model(data)

        # --- LOSS CALCULATION ---
        # Reconstruction Loss: How well the model reconstructs the input
        recon_loss = F.mse_loss(x_recon, data)

        # Perceptual Loss: Using LPIPS to measure perceptual similarity
        p_loss = perceptual_loss(x_recon, data)

        # 2. Total Loss: The sum of reconstruction loss and the VQ loss
        # The vq_loss already includes the commitment loss, as calculated inside the VectorQuantizer
        total_loss = recon_loss + vq_loss + (perceptual_loss_weight * p_loss)

        # --- BACKWARD PASS & OPTIMIZATION ---
        total_loss.backward()
        optimizer.step()

        # --- UPDATE METRICS ---
        total_train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_p_loss += p_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

    # --- LOGGING EPOCH RESULTS ---
    # Calculate average metrics for the entire epoch
    avg_train_loss = total_train_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_p_loss = total_p_loss / len(dataloader)
    avg_vq_loss = total_vq_loss / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    ema_perplexity = model.vq_layer.get_ema_perplexity()

    print(f'====> Epoch: {epoch} | Avg Loss: {avg_train_loss:.4f} | '
          f'Avg Recon Loss: {avg_recon_loss:.4f} '
          f'| Avg Perceptual Loss: {avg_p_loss:.4f} '
          f' | Avg VQ Loss: {avg_vq_loss:.4f} | '
          f'Avg Perplexity: {avg_perplexity:.2f}'
          f' | EMA Perplexity: {ema_perplexity:.2f}')

    # --- Resetting dead codes to random latents from the last batch ---
    if epoch > 1:
        model.vq_layer.reset_dead_codes(z)

    return avg_train_loss


def validate_vqvae(model, dataloader, device, perceptual_loss, perceptual_loss_weight=0.1):
    """
    Validates the VQ-VAE model on the validation set.
    """
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            x_recon, vq_loss, _, _, _, _ = model(data)
            recon_loss = F.mse_loss(x_recon, data)
            p_loss = perceptual_loss(x_recon, data)
            total_loss = recon_loss + vq_loss + (perceptual_loss_weight * p_loss)
            total_val_loss += total_loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    print(f'====> Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss


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

    # Split dataset into training and validation
    val_size = int(len(dataset) * config['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    # Initialize Model and Optimizer
    model = VQVAE().to(DEVICE)
    model.initialize_codebook(train_loader, DEVICE, num_batches_for_init=50)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    perceptual_loss = LPIPSLoss().to(DEVICE)

    # Training Loop
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, config["epochs"] + 1):
        train_vqvae_epoch(model, train_loader, optimizer, epoch, DEVICE, perceptual_loss,
                          perceptual_loss_weight=config["perceptual_loss_weight"])
        val_loss = validate_vqvae(model, val_loader, DEVICE, perceptual_loss,
                                  perceptual_loss_weight=config["perceptual_loss_weight"])

        if epoch % 5 == 0 or epoch == config["epochs"]:
            visualize_reconstruction(model, val_loader, DEVICE, epoch)

        if best_val_loss - val_loss > config["min_delta"]:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), VQ_VAE_CHECKPOINT_BEST_FILENAME)
            print(f"Validation loss improved. Saved model to {VQ_VAE_CHECKPOINT_BEST_FILENAME}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

        # print(torch.cuda.memory_summary(device=DEVICE, abbreviated=True))
    end_time = time.time()
    print(f"\nVAE Training finished in {end_time - start_time:.2f} seconds.")

    # Save the trained model
    try:
        torch.save(model.state_dict(), VQ_VAE_CHECKPOINT_FILENAME)  # Use path from utils
        print(f"Model saved to {VQ_VAE_CHECKPOINT_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")
