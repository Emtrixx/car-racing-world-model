# dream.py
import gymnasium as gym
import numpy as np
import torch
import time
import imageio
import matplotlib.pyplot as plt
import sys

from models.actor_critic import Actor
from train_world_model import GRU_HIDDEN_DIM, GRU_NUM_LAYERS, GRU_INPUT_EMBED_DIM
# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME, WM_CHECKPOINT_FILENAME, DREAM_GIF_FILENAME,
                   RandomPolicy, preprocess_and_encode, WM_CHECKPOINT_FILENAME_GRU, PPO_ACTOR_SAVE_FILENAME,
                   PPOPolicyWrapper)
from models.world_model import WorldModelMLP, WorldModelGRU
from models.conv_vae import ConvVAE

# --- Configuration ---
DREAM_HORIZON = 100 # How many steps to dream
DREAM_GIF_FILENAME = f"{ENV_NAME}_dream_gru_horizon{DREAM_HORIZON}.gif"

# --- Dreaming Function ---
def dream_sequence_gru(vae_model, world_model_gru, policy, initial_obs, transform_fn, horizon, device):
    vae_model.eval()
    world_model_gru.eval()
    dreamed_frames = []

    with torch.no_grad():
        # 1. Encode the initial observation
        z_current = preprocess_and_encode(initial_obs, transform_fn, vae_model, device) # (latent_dim)

        # 2. Decode the initial latent state to get the first frame
        initial_frame_decoded = vae_model.decode(z_current.unsqueeze(0)).squeeze(0)
        initial_frame_np = initial_frame_decoded.permute(1, 2, 0).cpu().numpy()
        dreamed_frames.append((np.clip(initial_frame_np, 0, 1) * 255).astype(np.uint8))

        print(f"Starting GRU dream. Horizon: {horizon}")

        # Initialize GRU hidden state (batch_size is 1 for dreaming)
        h_current = torch.zeros(world_model_gru.gru_num_layers, 1, world_model_gru.gru_hidden_dim).to(device)

        for t in range(horizon):
            # a. Get action from policy (expects (latent_dim))
            action_np = policy.get_action(z_current.cpu().numpy())
            action_t = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0) # (1, action_dim)
            z_current_batch = z_current.unsqueeze(0) # (1, latent_dim)

            # b. Predict next latent state using the GRU world model's step function
            z_next_pred, h_next = world_model_gru.step(z_current_batch, action_t, h_current)
            z_next_pred = z_next_pred.squeeze(0) # (latent_dim)

            # c. Decode the predicted latent state into an image
            obs_pred_decoded = vae_model.decode(z_next_pred.unsqueeze(0)).squeeze(0)

            # d. Store frame
            frame_np = obs_pred_decoded.permute(1, 2, 0).cpu().numpy()
            dreamed_frames.append((np.clip(frame_np, 0, 1) * 255).astype(np.uint8))

            # e. Update current latent state and hidden state
            z_current = z_next_pred
            h_current = h_next

            if (t + 1) % 20 == 0: print(f"  GRU Dream step {t+1}/{horizon}")
    print("GRU Dreaming finished.")
    return dreamed_frames

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting dream sequence generation on device: {DEVICE}")

    # 1. Initialize Environment (only for initial obs)
    temp_env = gym.make(ENV_NAME)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    action_space_low = temp_env.action_space.low
    action_space_high = temp_env.action_space.high
    temp_env.close()

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE) # Use definition from models
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError: print(f"ERROR: VAE ckpt '{VAE_CHECKPOINT_FILENAME}' not found."); env.close(); sys.exit()
    except Exception as e: print(f"ERROR loading VAE: {e}"); env.close(); sys.exit()

    # 3. Load GRU World Model
    world_model_gru = WorldModelGRU(
        gru_hidden_dim=GRU_HIDDEN_DIM,
        gru_num_layers=GRU_NUM_LAYERS,
        gru_input_embed_dim=GRU_INPUT_EMBED_DIM
    ).to(DEVICE)
    try:
        world_model_gru.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
        world_model_gru.eval()
        print(f"Loaded GRU World Model: {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e: print(f"ERROR GRU WM: {e}"); sys.exit()

    # 4. Initialize Policy (PPO Actor)
    policy_for_collection = None
    try:
        print(f"Attempting to load PPO Actor for dreaming: {PPO_ACTOR_SAVE_FILENAME}")
        actor_for_dreaming = Actor().to(DEVICE)
        actor_for_dreaming.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
        actor_for_dreaming.eval()
        # For dreaming, deterministic actions might be preferred for consistency
        policy_for_dreaming = PPOPolicyWrapper(actor_for_dreaming, DEVICE, deterministic=True,
                                               action_space_low=action_space_low,
                                               action_space_high=action_space_high)
        print(f"Using PPO Actor for data collection.")
    except FileNotFoundError:
        print(f"ERROR: PPO Actor checkpoint '{PPO_ACTOR_SAVE_FILENAME}' not found. Train PPO first.")
        env.close();
        sys.exit()
    except Exception as e:
        print(f"ERROR loading PPO Actor: {e}");
        env.close();
        sys.exit()

    # Get Initial Observation for dreaming
    # No need to keep env open if just getting one frame from a fresh reset
    dream_env = gym.make(ENV_NAME, render_mode="rgb_array")
    initial_obs, _ = dream_env.reset()
    dream_env.close()

    start_dream_time = time.time()
    dreamed_frames = dream_sequence_gru(
        vae_model, world_model_gru, policy_for_dreaming, initial_obs, transform, DREAM_HORIZON, DEVICE
    )
    print(f"GRU Dream generation took {time.time() - start_dream_time:.2f} seconds.")

    # 7. Save GIF
    if dreamed_frames:
        try:
            print(f"Saving dream GIF ({len(dreamed_frames)} frames) to {DREAM_GIF_FILENAME}...")
            imageio.mimsave(DREAM_GIF_FILENAME, dreamed_frames, fps=15) # Adjust fps
            print("GIF saved.")
        except Exception as e: print(f"Error saving GIF: {e}")

        # 8. Save Preview Plot
        try:
            num_display = min(len(dreamed_frames), 10)
            fig, axes = plt.subplots(1, num_display, figsize=(num_display * 1.5, 1.5))
            fig.suptitle(f'Dream Sequence Preview (First {num_display} Frames)', fontsize=12)
            for i in range(num_display):
                 axes[i].imshow(dreamed_frames[i]); axes[i].set_title(f'T={i}'); axes[i].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            preview_path = "images/dream_sequence_preview.png"
            plt.savefig(preview_path); print(f"Saved preview plot to {preview_path}"); plt.close(fig)
        except Exception as e: print(f"Error saving preview plot: {e}")