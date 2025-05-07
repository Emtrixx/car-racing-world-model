# train_world_model.py
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

from models.actor_critic import Actor
from models.conv_vae import ConvVAE
from models.world_model import WorldModelGRU
from utils import WM_CHECKPOINT_FILENAME_GRU
# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME,  # WM_CHECKPOINT_FILENAME (will change suffix)
                   preprocess_and_encode, PPO_ACTOR_SAVE_FILENAME, PPOPolicyWrapper)

# --- Configuration ---
WM_LEARNING_RATE = 1e-4
WM_EPOCHS = 50 # Might need more for GRU
WM_BATCH_SIZE = 32 # Sequences per batch
COLLECT_EPISODES = 350 # Number of full episodes to collect for WM training
REPLAY_BUFFER_CAPACITY = COLLECT_EPISODES # Store sequences from episodes

# GRU Specific Config
GRU_HIDDEN_DIM = 256
GRU_NUM_LAYERS = 1 # Start with 1, can try 2
GRU_INPUT_EMBED_DIM = 128 # Optional embedding dimension for (z,a) pair
SEQUENCE_LENGTH = 50  # Length of sequences to train on


# --- Data Collection for GRU (Sequence Data) ---
def collect_sequences_for_gru(env, policy, transform_fn, vae_model,
                              num_episodes, sequence_length, device):
    print(f"Collecting sequences from {num_episodes} episodes for GRU World Model...")
    collected_sequences = [] # Store tuples of (z_input_seq, a_input_seq, z_target_seq)

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        current_episode_z = []
        current_episode_a = []
        done = False
        truncated = False

        # Collect full episode
        while not done and not truncated:
            z_t = preprocess_and_encode(obs, transform_fn, vae_model, device)
            action_np = policy.get_action(z_t.cpu().numpy()) # Policy might expect numpy
            action_tensor = torch.tensor(action_np, dtype=torch.float32)

            current_episode_z.append(z_t.cpu())       # Store z_t
            current_episode_a.append(action_tensor.cpu()) # Store a_t

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

        # Add the final z state if episode ended
        if len(current_episode_z) > 0: # Ensure episode was not empty
            z_final = preprocess_and_encode(obs, transform_fn, vae_model, device)
            current_episode_z.append(z_final.cpu()) # This is z_L (target for last input)

        # Create fixed-length sequences from the episode
        # We need sequence_length+1 z's to make sequence_length inputs and sequence_length targets
        episode_len = len(current_episode_a) # Number of actions, so L-1 inputs
        if episode_len >= sequence_length:
            for i in range(episode_len - sequence_length + 1):
                # z_inputs: z_i, z_{i+1}, ..., z_{i+sequence_length-1}
                z_input_seq = torch.stack(current_episode_z[i : i + sequence_length])
                # a_inputs: a_i, a_{i+1}, ..., a_{i+sequence_length-1}
                a_input_seq = torch.stack(current_episode_a[i : i + sequence_length])
                # z_targets: z_{i+1}, z_{i+2}, ..., z_{i+sequence_length}
                z_target_seq = torch.stack(current_episode_z[i + 1 : i + 1 + sequence_length])

                collected_sequences.append((z_input_seq, a_input_seq, z_target_seq))
        if (episode_idx + 1) % 20 == 0:
            print(f"  Finished episode {episode_idx+1}/{num_episodes}. Total sequences: {len(collected_sequences)}")

    print(f"Finished collecting. Total sequences: {len(collected_sequences)}.")
    return collected_sequences


# --- Dataset for GRU World Model Training (Sequences) ---
class SequenceDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data # List of (z_input_seq, a_input_seq, z_target_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] # Returns the tuple of three sequence tensors

# --- GRU World Model Training Loop ---
def train_world_model_gru_epoch(world_model_gru, dataloader, optimizer, criterion, epoch, device):
    world_model_gru.train()
    epoch_loss = 0
    processed_batches = 0

    for batch_idx, (z_input_seq, a_input_seq, z_target_seq) in enumerate(dataloader):
        z_input_seq = z_input_seq.to(device)     # (batch, seq_len, latent_dim)
        a_input_seq = a_input_seq.to(device)     # (batch, seq_len, action_dim)
        z_target_seq = z_target_seq.to(device)   # (batch, seq_len, latent_dim)

        optimizer.zero_grad()

        # Predict sequence of next latent states
        # h_initial is None by default (zeros)
        z_pred_seq, _ = world_model_gru(z_input_seq, a_input_seq)

        # Calculate loss over the entire sequence predictions
        loss = criterion(z_pred_seq, z_target_seq)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        processed_batches += 1

    avg_epoch_loss = epoch_loss / processed_batches
    print(f'====> GRU World Model Epoch: {epoch} Average loss: {avg_epoch_loss:.6f}')
    return avg_epoch_loss

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting GRU World Model training on device: {DEVICE}")

    # 1. Initialize Environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        env.close(); exit()
    except Exception as e:
        print(f"ERROR loading VAE: {e}"); env.close(); exit()

    # 3. Initialize Policy (PPO Actor)
    policy_for_collection = None
    try:
        print(f"Attempting to load PPO Actor from: {PPO_ACTOR_SAVE_FILENAME}")
        actor_for_collection = Actor().to(DEVICE)  # Actor from models.py
        actor_for_collection.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
        actor_for_collection.eval()  # Set to eval mode
        # For data collection for world model, sampling is often preferred (deterministic=False)
        policy_for_collection = PPOPolicyWrapper(actor_for_collection, DEVICE, deterministic=False,
                                                 action_space_low=env.action_space.low,
                                                 action_space_high=env.action_space.high)
        print(f"Using PPO Actor for data collection.")
    except FileNotFoundError:
        print(f"ERROR: PPO Actor checkpoint '{PPO_ACTOR_SAVE_FILENAME}' not found. Train PPO first.")
        env.close(); exit()
    except Exception as e:
        print(f"ERROR loading PPO Actor: {e}"); env.close(); exit()

    # 4. Collect Sequence Data
    start_collect_time = time.time()
    sequence_data_buffer = collect_sequences_for_gru(
        env, policy_for_collection, transform, vae_model,
        COLLECT_EPISODES, SEQUENCE_LENGTH, DEVICE
    )
    env.close()
    print(f"Sequence data collection took {time.time() - start_collect_time:.2f} seconds.")

    if not sequence_data_buffer:
        print("ERROR: No sequence data collected. Exiting.")
        exit()

    # 5. Prepare DataLoader
    sequence_dataset = SequenceDataset(sequence_data_buffer)
    wm_dataloader = DataLoader(sequence_dataset, batch_size=WM_BATCH_SIZE, shuffle=True)

    # 6. Initialize GRU World Model, Optimizer, Criterion
    world_model_gru = WorldModelGRU(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        gru_num_layers=GRU_NUM_LAYERS,
        gru_input_embed_dim=GRU_INPUT_EMBED_DIM
    ).to(DEVICE)
    wm_optimizer = optim.Adam(world_model_gru.parameters(), lr=WM_LEARNING_RATE)
    wm_criterion = nn.MSELoss() # Mean Squared Error for predicting next latent state sequence

    # 7. Train the GRU World Model
    print("Starting GRU World Model training...")
    start_train_time = time.time()
    wm_losses = []
    for epoch in range(1, WM_EPOCHS + 1):
        loss = train_world_model_gru_epoch(world_model_gru, wm_dataloader, wm_optimizer, wm_criterion, epoch, DEVICE)
        wm_losses.append(loss)
    print(f"GRU World Model training took {time.time() - start_train_time:.2f} seconds.")

    # 8. Plot and Save Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, WM_EPOCHS + 1), wm_losses)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"GRU World Model Training Loss (SeqLen {SEQUENCE_LENGTH})"); plt.grid(True)
    loss_plot_path = f"images/world_model_gru_loss_seq{SEQUENCE_LENGTH}.png"
    plt.savefig(loss_plot_path); print(f"Saved loss plot to {loss_plot_path}"); plt.close()

    # 9. Save the trained GRU World Model
    try:
        torch.save(world_model_gru.state_dict(), WM_CHECKPOINT_FILENAME_GRU)
        print(f"GRU World Model saved to {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e:
        print(f"Error saving GRU World Model: {e}")