# train_world_model.py
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

from src.legacy.actor_critic import Actor
from src.legacy.conv_vae import ConvVAE
from src.world_model import WorldModelGRU
from src.utils import WM_CHECKPOINT_FILENAME_GRU, FrameStackWrapper, NUM_STACK, IMAGES_DIR
# Import from local modules
from src.utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM,
                       )
from src.legacy.utils_legacy import transform, preprocess_and_encode, preprocess_and_encode_stack, \
    VAE_CHECKPOINT_FILENAME
from src.legacy.utils_rl import PPO_ACTOR_SAVE_FILENAME, PPOPolicyWrapper

# --- Configuration ---
WM_LEARNING_RATE = 1e-4
WM_EPOCHS = 10  # Might need more for GRU
# WM_EPOCHS = 100  # Might need more for GRU
WM_BATCH_SIZE = 32  # Sequences per batch
COLLECT_EPISODES = 10  # Number of full episodes to collect for WM training
# COLLECT_EPISODES = 500  # Number of full episodes to collect for WM training
REPLAY_BUFFER_CAPACITY = COLLECT_EPISODES  # Store sequences from episodes

# GRU Specific Config
GRU_HIDDEN_DIM = 256
GRU_NUM_LAYERS = 2  # Start with 1, can try 2
GRU_INPUT_EMBED_DIM = 128  # Optional embedding dimension for (z,a) pair
SEQUENCE_LENGTH = 50  # Length of sequences to train on


# --- Data Collection for GRU (Sequence Data) ---
def collect_sequences_for_gru(env, policy, transform_fn, vae_model,
                              num_episodes, sequence_length, device):
    print(f"Collecting sequences (z,a,r,d) from {num_episodes} episodes...")
    collected_sequences = []

    for episode_idx in range(num_episodes):
        # obs_stack_raw is (NUM_STACK, H, W, C) from FrameStackWrapper
        obs_stack_raw, _ = env.reset()

        # For storing data of the current episode before forming sequences
        ep_Z_input_stacks_cpu = []  # List of Z_t (concatenated stack, NUM_STACK * LATENT_DIM)
        ep_a_actions_cpu = []  # List of a_t (ACTION_DIM)
        ep_r_rewards_cpu = []  # List of r_{t+1} (scalar)
        ep_d_dones_cpu = []  # List of d_{t+1} (scalar)
        ep_z_single_targets_cpu = []  # List of z'_{t+1} (single frame latent, LATENT_DIM)

        done, truncated = False, False
        current_steps_in_episode = 0

        while not done and not truncated:
            # 1. Current state Z_t (input for GRU) - This is the stack of latents for current time t
            Z_t_concat_gpu = preprocess_and_encode_stack(obs_stack_raw, transform_fn, vae_model, device)
            ep_Z_input_stacks_cpu.append(Z_t_concat_gpu.cpu())

            # 2. Get action based on Z_t
            action_np = policy.get_action(Z_t_concat_gpu.cpu().numpy())  # Policy sees the concatenated stack
            action_tensor = torch.tensor(action_np, dtype=torch.float32)
            ep_a_actions_cpu.append(action_tensor.cpu())

            # 3. Step environment
            next_obs_stack_raw, reward, terminated, truncated, info = env.step(action_np)
            current_done_flag = terminated or truncated

            # 4. Prepare targets for this step (r_{t+1}, d_{t+1}, and z'_{t+1})
            ep_r_rewards_cpu.append(torch.tensor(reward, dtype=torch.float32).cpu())
            ep_d_dones_cpu.append(torch.tensor(current_done_flag, dtype=torch.float32).cpu())

            # The target z'_{t+1} is the VAE encoding of the *single newest raw frame*
            # from next_obs_stack_raw. The newest frame is at the last index of the stack.
            single_newest_raw_frame = next_obs_stack_raw[-1]  # Shape (H, W, C)
            z_prime_tp1_gpu = preprocess_and_encode(single_newest_raw_frame, transform_fn, vae_model, device)
            ep_z_single_targets_cpu.append(z_prime_tp1_gpu.cpu())

            # 5. Update for next iteration
            obs_stack_raw = next_obs_stack_raw
            done = current_done_flag
            current_steps_in_episode += 1

        # After episode ends, create sequences of fixed length
        num_transitions_in_episode = len(ep_a_actions_cpu)  # This is L, number of (S,A,R,S',D) steps

        if num_transitions_in_episode >= sequence_length:
            for i in range(num_transitions_in_episode - sequence_length + 1):
                # Input Z_stack sequence: Z_i, Z_{i+1}, ..., Z_{i+sequence_length-1}
                z_input_s = torch.stack(ep_Z_input_stacks_cpu[i: i + sequence_length])
                # Action sequence: a_i, ..., a_{i+sequence_length-1}
                a_input_s = torch.stack(ep_a_actions_cpu[i: i + sequence_length])

                # Target single z' sequence: z'_{i+1}, ..., z'_{i+sequence_length}
                # These correspond to the single next frame's latent for each input Z_stack
                z_target_s = torch.stack(ep_z_single_targets_cpu[i: i + sequence_length])
                # Target reward sequence: r_{i+1}, ..., r_{i+sequence_length}
                r_target_s = torch.stack(ep_r_rewards_cpu[i: i + sequence_length]).unsqueeze(-1)
                # Target done sequence: d_{i+1}, ..., d_{i+sequence_length}
                d_target_s = torch.stack(ep_d_dones_cpu[i: i + sequence_length]).unsqueeze(-1)

                collected_sequences.append((z_input_s, a_input_s, z_target_s, r_target_s, d_target_s))

        if (episode_idx + 1) % 20 == 0 or episode_idx == num_episodes - 1:
            print(
                f"  Episode {episode_idx + 1}/{num_episodes}. Steps: {current_steps_in_episode}. Total sequences collected: {len(collected_sequences)}")
    print(f"Finished collecting. Total sequences: {len(collected_sequences)}.")
    return collected_sequences


class SequenceDataset(Dataset):  # Now returns 5 items
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


# --- GRU World Model Training Loop (with r, d loss) ---
def train_world_model_gru_epoch(world_model_gru, dataloader, optimizer,
                                z_criterion, r_criterion, d_criterion, epoch, device):  # Separate criteria
    world_model_gru.train()
    epoch_loss, epoch_z_loss, epoch_r_loss, epoch_d_loss = 0, 0, 0, 0
    processed_batches = 0

    for z_input_seq, a_input_seq, z_target_seq, r_target_seq, d_target_seq in dataloader:
        z_input_seq, a_input_seq, z_target_seq, r_target_seq, d_target_seq = \
            z_input_seq.to(device), a_input_seq.to(device), z_target_seq.to(device), \
                r_target_seq.to(device), d_target_seq.to(device)

        optimizer.zero_grad()
        z_pred_seq, r_pred_seq, d_pred_logits_seq, _ = world_model_gru(z_input_seq, a_input_seq)

        loss_z = z_criterion(z_pred_seq, z_target_seq)
        loss_r = r_criterion(r_pred_seq, r_target_seq)
        # Use BCEWithLogitsLoss for done flags as they are binary
        loss_d = d_criterion(d_pred_logits_seq, d_target_seq)  # Target should be float 0.0 or 1.0

        # Combine losses (can weight them)
        total_loss = loss_z + loss_r + loss_d

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_z_loss += loss_z.item()
        epoch_r_loss += loss_r.item()
        epoch_d_loss += loss_d.item()
        processed_batches += 1

    avg_loss = epoch_loss / processed_batches
    avg_z_loss = epoch_z_loss / processed_batches
    avg_r_loss = epoch_r_loss / processed_batches
    avg_d_loss = epoch_d_loss / processed_batches
    print(f'====> GRU WM Epoch: {epoch} Avg Loss: {avg_loss:.4f} '
          f'(Z: {avg_z_loss:.4f}, R: {avg_r_loss:.4f}, D: {avg_d_loss:.4f})')
    return avg_loss


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting GRU World Model training on device: {DEVICE}")

    # 1. Initialize Environment
    env = FrameStackWrapper(gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=400), num_stack=NUM_STACK)

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        env.close();
        exit()
    except Exception as e:
        print(f"ERROR loading VAE: {e}");
        env.close();
        exit()

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
        env.close();
        exit()
    except Exception as e:
        print(f"ERROR loading PPO Actor: {e}");
        env.close();
        exit()

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
        latent_dim=LATENT_DIM,  # Input is the concatenated stack
        action_dim=ACTION_DIM,
    ).to(DEVICE)
    wm_optimizer = optim.Adam(world_model_gru.parameters(), lr=WM_LEARNING_RATE)
    # Define separate loss criteria
    z_loss_criterion = nn.MSELoss()
    r_loss_criterion = nn.MSELoss()  # Or nn.L1Loss()
    d_loss_criterion = nn.BCEWithLogitsLoss()  # For done logits

    # 7. Train the GRU World Model
    print("Starting GRU World Model training...")
    start_train_time = time.time()
    wm_losses = []
    for epoch in range(1, WM_EPOCHS + 1):
        loss = train_world_model_gru_epoch(world_model_gru, wm_dataloader, wm_optimizer,
                                           z_loss_criterion, r_loss_criterion, d_loss_criterion,
                                           epoch, DEVICE)
        wm_losses.append(loss)
    print(f"GRU World Model training took {time.time() - start_train_time:.2f} seconds.")

    # 8. Plot and Save Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, WM_EPOCHS + 1), wm_losses)
    plt.xlabel("Epoch");
    plt.ylabel("MSE Loss")
    plt.title(f"GRU World Model Training Loss (SeqLen {SEQUENCE_LENGTH})");
    plt.grid(True)
    loss_plot_path = IMAGES_DIR / f"world_model_gru_loss_seq{SEQUENCE_LENGTH}.png"
    plt.savefig(loss_plot_path);
    print(f"Saved loss plot to {loss_plot_path}");
    plt.close()

    # 9. Save the trained GRU World Model
    try:
        torch.save(world_model_gru.state_dict(), WM_CHECKPOINT_FILENAME_GRU)
        print(f"GRU World Model saved to {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e:
        print(f"Error saving GRU World Model: {e}")
