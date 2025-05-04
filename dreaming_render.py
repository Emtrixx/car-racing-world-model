# dream.py
import gymnasium as gym
import numpy as np
import torch
import time
import imageio
import matplotlib.pyplot as plt
import sys

# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME, WM_CHECKPOINT_FILENAME, DREAM_GIF_FILENAME,
                   RandomPolicy, preprocess_and_encode)
from models import ConvVAE, WorldModelMLP

# --- Configuration ---
DREAM_HORIZON = 100 # How many steps to dream

# --- Dreaming Function ---
def dream_sequence(vae_model, world_model, policy, initial_obs, transform_fn, horizon, device):
    vae_model.eval(); world_model.eval()
    dreamed_frames = []

    with torch.no_grad():
        z_current = preprocess_and_encode(initial_obs, transform_fn, vae_model, device)
        initial_frame_decoded = vae_model.decode(z_current.unsqueeze(0)).squeeze(0)
        initial_frame_np = initial_frame_decoded.permute(1, 2, 0).cpu().numpy()
        dreamed_frames.append((np.clip(initial_frame_np, 0, 1) * 255).astype(np.uint8))

        print(f"Starting dream. Horizon: {horizon}")
        pbar_interval = max(1, horizon // 10)

        for t in range(horizon):
            action_np = policy.get_action(z_current.cpu().numpy())
            action = torch.tensor(action_np, dtype=torch.float32, device=device)

            # Predict next latent state (World Model expects batch dim)
            z_next_pred = world_model(z_current.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

            # Decode predicted latent state (VAE decoder expects batch dim)
            obs_pred_decoded = vae_model.decode(z_next_pred.unsqueeze(0)).squeeze(0)

            # Store frame
            frame_np = obs_pred_decoded.permute(1, 2, 0).cpu().numpy()
            dreamed_frames.append((np.clip(frame_np, 0, 1) * 255).astype(np.uint8))

            # Update state
            z_current = z_next_pred

            if (t + 1) % pbar_interval == 0: print(f"  Dream step {t+1}/{horizon}")

    print("Dreaming finished.")
    return dreamed_frames

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting dream sequence generation on device: {DEVICE}")

    # 1. Initialize Environment (only for initial obs)
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE) # Use definition from models
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError: print(f"ERROR: VAE ckpt '{VAE_CHECKPOINT_FILENAME}' not found."); env.close(); sys.exit()
    except Exception as e: print(f"ERROR loading VAE: {e}"); env.close(); sys.exit()

    # 3. Load Trained World Model
    world_model = WorldModelMLP().to(DEVICE) # Use definition from models
    try:
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME, map_location=DEVICE))
        world_model.eval()
        print(f"Loaded World Model: {WM_CHECKPOINT_FILENAME}")
    except FileNotFoundError: print(f"ERROR: WM ckpt '{WM_CHECKPOINT_FILENAME}' not found."); env.close(); sys.exit()
    except Exception as e: print(f"ERROR loading WM: {e}"); env.close(); sys.exit()

    # 4. Initialize Policy (Random for testing dream)
    policy = RandomPolicy(env.action_space) # From utils
    # Replace with your trained policy later if desired

    # 5. Get Initial Observation
    initial_obs, _ = env.reset()
    env.close() # Close env

    # 6. Generate Dream Sequence
    start_dream_time = time.time()
    # Pass transform from utils
    dreamed_frames = dream_sequence(vae_model, world_model, policy, initial_obs, transform, DREAM_HORIZON, DEVICE)
    print(f"Dream generation took {time.time() - start_dream_time:.2f} seconds.")

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