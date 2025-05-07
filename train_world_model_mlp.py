# train_world_model_mlp.py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import deque
import time
import matplotlib.pyplot as plt

# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME, WM_CHECKPOINT_FILENAME,
                   RandomPolicy, preprocess_and_encode)
from projects.gym_stuff.car_racing.models.world_model import WorldModelMLP
from projects.gym_stuff.car_racing.models.conv_vae import ConvVAE

# --- Configuration ---
WM_LEARNING_RATE = 1e-4
WM_EPOCHS = 20
WM_BATCH_SIZE = 64
COLLECT_STEPS = 10000 # Number of environment steps for WM training data
REPLAY_BUFFER_CAPACITY = COLLECT_STEPS

# --- Data Collection with Latent States ---
def collect_latent_transitions(env, policy, transform_fn, vae_model, replay_buffer, num_steps, device):
    print(f"Collecting {num_steps} transitions for World Model training...")
    obs, _ = env.reset()
    # Encode initial state (using helper from utils)
    z_current = preprocess_and_encode(obs, transform_fn, vae_model, device)
    collected_count = 0

    pbar_interval = max(1, num_steps // 20) # Print progress roughly 20 times

    while collected_count < num_steps:
        # Use policy (RandomPolicy from utils for now)
        action_np = policy.get_action(z_current.cpu().numpy())
        action = torch.tensor(action_np, dtype=torch.float32) # Keep action as tensor

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        collected_count += 1

        # Encode next state
        z_next = preprocess_and_encode(next_obs, transform_fn, vae_model, device)

        # Store transition (z_t, a_t, z_{t+1}) - storing tensors on CPU
        transition = (z_current.cpu(), action.cpu(), z_next.cpu())
        replay_buffer.append(transition)

        # Update state
        z_current = z_next
        if done:
            obs, _ = env.reset()
            z_current = preprocess_and_encode(obs, transform_fn, vae_model, device)

        if collected_count % pbar_interval == 0:
            print(f"  Collected {collected_count}/{num_steps} transitions...")

    print(f"Finished collecting {len(replay_buffer)} transitions.")
    return replay_buffer

# --- Dataset for World Model Training ---
class TransitionDataset(Dataset):
    def __init__(self, buffer):
        self.data = list(buffer) # Convert deque to list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns z_t, a_t, z_tp1
        return self.data[idx]

# --- World Model Training Loop ---
def train_world_model_epoch(world_model, dataloader, optimizer, criterion, epoch, device):
    world_model.train()
    epoch_loss = 0
    processed_batches = 0

    for batch_idx, (z_t, a_t, z_tp1) in enumerate(dataloader):
        z_t, a_t, z_tp1 = z_t.to(device), a_t.to(device), z_tp1.to(device)
        optimizer.zero_grad()
        z_tp1_pred = world_model(z_t, a_t)
        loss = criterion(z_tp1_pred, z_tp1)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        processed_batches += 1

    avg_epoch_loss = epoch_loss / processed_batches
    print(f'====> World Model Epoch: {epoch} Average loss: {avg_epoch_loss:.6f}')
    return avg_epoch_loss

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting World Model training on device: {DEVICE}")

    # 1. Initialize Environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE) # Use definition from models
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        env.close(); exit()
    except Exception as e:
        print(f"ERROR loading VAE: {e}"); env.close(); exit()

    # 3. Initialize Policy (Random for data collection)
    policy = RandomPolicy(env.action_space) # From utils

    # 4. Initialize Replay Buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)

    # 5. Collect Data
    start_collect_time = time.time()
    # Pass transform from utils
    replay_buffer = collect_latent_transitions(env, policy, transform, vae_model, replay_buffer, COLLECT_STEPS, DEVICE)
    env.close()
    print(f"Data collection took {time.time() - start_collect_time:.2f} seconds.")

    if not replay_buffer:
        print("ERROR: No data collected. Exiting.")
        exit()

    # 6. Prepare DataLoader
    transition_dataset = TransitionDataset(replay_buffer)
    wm_dataloader = DataLoader(transition_dataset, batch_size=WM_BATCH_SIZE, shuffle=True)

    # 7. Initialize World Model, Optimizer, Criterion
    world_model = WorldModelMLP().to(DEVICE) # Use definition from models
    wm_optimizer = optim.Adam(world_model.parameters(), lr=WM_LEARNING_RATE)
    wm_criterion = nn.MSELoss()

    # 8. Train the World Model
    print("Starting World Model training...")
    start_train_time = time.time()
    wm_losses = []
    for epoch in range(1, WM_EPOCHS + 1):
        loss = train_world_model_epoch(world_model, wm_dataloader, wm_optimizer, wm_criterion, epoch, DEVICE)
        wm_losses.append(loss)
    print(f"World Model training took {time.time() - start_train_time:.2f} seconds.")

    # 9. Plot and Save Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, WM_EPOCHS + 1), wm_losses)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("World Model Training Loss ($z_{t+1}$ prediction)"); plt.grid(True)
    loss_plot_path = "images/world_model_training_loss.png"
    plt.savefig(loss_plot_path); print(f"Saved loss plot to {loss_plot_path}"); plt.close()


    # 10. Save the trained World Model
    try:
        torch.save(world_model.state_dict(), WM_CHECKPOINT_FILENAME) # Use path from utils
        print(f"World Model saved to {WM_CHECKPOINT_FILENAME}")
    except Exception as e:
        print(f"Error saving World Model: {e}")