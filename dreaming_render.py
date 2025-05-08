# dream.py
import gymnasium as gym
import numpy as np
import torch
import time
import imageio
import matplotlib.pyplot as plt
import sys # For sys.exit()

from projects.gym_stuff.car_racing.models.actor_critic import Actor
from projects.gym_stuff.car_racing.models.conv_vae import ConvVAE
from projects.gym_stuff.car_racing.models.world_model import WorldModelGRU
# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode)
from projects.gym_stuff.car_racing.utils_rl import PPO_ACTOR_SAVE_FILENAME, RandomPolicy, PPOPolicyWrapper

# Assuming GRU hyperparams and checkpoint name are correctly sourced.
# If they are defined in train_world_model.py at global scope:
try:
    from train_world_model import (WM_CHECKPOINT_FILENAME_GRU, GRU_HIDDEN_DIM,
                                   GRU_NUM_LAYERS, GRU_INPUT_EMBED_DIM)
except ImportError:
    print("ERROR: Could not import GRU params from train_world_model.py.")
    print("Please ensure it's in PYTHONPATH and defines these constants globally, or define them in utils.py / here.")
    # Define fallbacks or exit if critical
    # Example fallbacks (ensure these match your trained model if import fails)
    # GRU_HIDDEN_DIM = 256
    # GRU_NUM_LAYERS = 1
    # GRU_INPUT_EMBED_DIM = 128
    # WM_CHECKPOINT_FILENAME_GRU = f"{ENV_NAME}_worldmodel_gru_ld{LATENT_DIM}_ac{ACTION_DIM}_gru{GRU_HIDDEN_DIM}x{GRU_NUM_LAYERS}_seqLENGTH.pth" # Placeholder
    sys.exit(1)


# --- Configuration ---
DREAM_HORIZON = 100 # How many steps to dream
DREAM_GIF_FILENAME = f"images/{ENV_NAME}_dream_gru_horizon{DREAM_HORIZON}.gif" # Make sure 'images' dir exists

# --- Dreaming Function ---
def dream_sequence_gru(vae_model, world_model_gru, policy, initial_obs, transform_fn, horizon, device):
    vae_model.eval()
    world_model_gru.eval()
    dreamed_frames = []
    total_predicted_reward_in_dream = 0.0
    predicted_done_flag_raised = False # To stop if model predicts done

    with torch.no_grad():
        # 1. Encode the initial observation
        z_current = preprocess_and_encode(initial_obs, transform_fn, vae_model, device)

        # 2. Decode the initial latent state to get the first frame
        initial_frame_decoded = vae_model.decode(z_current.unsqueeze(0)).squeeze(0)
        initial_frame_np = initial_frame_decoded.permute(1, 2, 0).cpu().numpy()
        dreamed_frames.append((np.clip(initial_frame_np, 0, 1) * 255).astype(np.uint8))

        print(f"Starting GRU dream. Horizon: {horizon}")
        print(f"Step 0: Initial Frame. z_norm: {z_current.norm().item():.2f}")

        h_current = torch.zeros(world_model_gru.gru_num_layers, 1, world_model_gru.gru_hidden_dim).to(device)

        for t in range(horizon):
            if predicted_done_flag_raised and t > 0: # Stop if predicted done in previous step
                print(f"  Dream step {t}: Predicted 'done' in previous step. Ending dream early.")
                break

            action_np = policy.get_action(z_current.cpu().numpy())
            action_t = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
            z_current_batch = z_current.unsqueeze(0)

            # b. Predict next latent state, reward, and done using the GRU world model's step function
            # WorldModelGRU.step now returns: next_z_pred, next_r_pred, next_d_pred_logits, h_next
            next_z_pred, r_pred, d_logit_pred, h_next = \
                world_model_gru.step(z_current_batch, action_t, h_current)

            next_z_pred = next_z_pred.squeeze(0)
            predicted_reward_scalar = r_pred.squeeze().item() # Get scalar value
            predicted_done_prob = torch.sigmoid(d_logit_pred.squeeze()).item()

            total_predicted_reward_in_dream += predicted_reward_scalar

            # Print predicted reward and other info
            print(f"  Dream step {t+1}/{horizon}: Pred_R: {predicted_reward_scalar:.4f}, "
                  f"Pred_Done_Prob: {predicted_done_prob:.4f}, "
                  f"Next_z_norm: {next_z_pred.norm().item():.2f}")

            obs_pred_decoded = vae_model.decode(next_z_pred.unsqueeze(0)).squeeze(0)
            frame_np = obs_pred_decoded.permute(1, 2, 0).cpu().numpy()
            dreamed_frames.append((np.clip(frame_np, 0, 1) * 255).astype(np.uint8))

            z_current = next_z_pred
            h_current = h_next

            # Optional: End dream early if world model predicts "done" with high confidence
            if predicted_done_prob > 0.95: # Higher threshold for more confident "done"
                predicted_done_flag_raised = True


    print(f"GRU Dreaming finished. Total predicted reward in dream: {total_predicted_reward_in_dream:.2f}")
    return dreamed_frames

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting dream sequence generation on device: {DEVICE}")

    # Ensure images directory exists
    import os
    os.makedirs("images", exist_ok=True)

    # Initialize temp env for action space details for PPOPolicyWrapper
    temp_env = gym.make(ENV_NAME)
    action_space_low = temp_env.action_space.low
    action_space_high = temp_env.action_space.high
    temp_env.close()

    # Load VAE
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError: print(f"ERROR: VAE ckpt '{VAE_CHECKPOINT_FILENAME}' not found."); sys.exit()
    except Exception as e: print(f"ERROR loading VAE: {e}"); sys.exit()

    # Load GRU World Model
    # Ensure GRU_HIDDEN_DIM, GRU_NUM_LAYERS, GRU_INPUT_EMBED_DIM are available
    world_model_gru = WorldModelGRU(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, # From utils
        gru_hidden_dim=GRU_HIDDEN_DIM,
        gru_num_layers=GRU_NUM_LAYERS,
        gru_input_embed_dim=GRU_INPUT_EMBED_DIM
    ).to(DEVICE)
    try:
        world_model_gru.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
        world_model_gru.eval()
        print(f"Loaded GRU World Model: {WM_CHECKPOINT_FILENAME_GRU}")
    except FileNotFoundError: print(f"ERROR: GRU WM ckpt '{WM_CHECKPOINT_FILENAME_GRU}' not found."); sys.exit()
    except Exception as e: print(f"ERROR loading GRU WM: {e}"); sys.exit()

    # Initialize Policy for Dreaming (Load PPO Actor or fallback to Random)
    policy_for_dreaming = None
    try:
        print(f"Attempting to load PPO Actor for dreaming: {PPO_ACTOR_SAVE_FILENAME}")
        actor_for_dreaming = Actor().to(DEVICE)
        actor_for_dreaming.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
        actor_for_dreaming.eval()
        policy_for_dreaming = PPOPolicyWrapper(actor_for_dreaming, DEVICE, deterministic=True,
                                               action_space_low=action_space_low,
                                               action_space_high=action_space_high)
        print(f"Using PPO Actor for dreaming.")
    except FileNotFoundError:
        print(f"PPO Actor model not found at '{PPO_ACTOR_SAVE_FILENAME}'. Using RandomPolicy for dreaming.")
        env_action_space = gym.make(ENV_NAME).action_space
        policy_for_dreaming = RandomPolicy(env_action_space)
    except Exception as e:
        print(f"Error loading PPO Actor, using RandomPolicy: {e}")
        env_action_space = gym.make(ENV_NAME).action_space
        policy_for_dreaming = RandomPolicy(env_action_space)

    # Get Initial Observation for dreaming
    dream_env = gym.make(ENV_NAME, render_mode="rgb_array")
    initial_obs, _ = dream_env.reset()
    dream_env.close()

    start_dream_time = time.time()
    dreamed_frames = dream_sequence_gru(
        vae_model, world_model_gru, policy_for_dreaming, initial_obs, transform, DREAM_HORIZON, DEVICE
    )
    print(f"GRU Dream generation took {time.time() - start_dream_time:.2f} seconds.")

    if dreamed_frames:
        try:
            print(f"Saving GRU dream GIF ({len(dreamed_frames)} frames) to {DREAM_GIF_FILENAME}...")
            imageio.mimsave(DREAM_GIF_FILENAME, dreamed_frames, fps=15)
            print("GIF saved.")
        except Exception as e: print(f"Error saving GIF: {e}")

        try:
            num_display = min(len(dreamed_frames), 10)
            fig, axes = plt.subplots(1, num_display, figsize=(num_display * 1.5, 1.5))
            if num_display == 1: axes = [axes] # Make it iterable if only one subplot
            fig.suptitle(f'Dream Sequence Preview (First {num_display} Frames)', fontsize=12)
            for i in range(num_display):
                 axes[i].imshow(dreamed_frames[i]); axes[i].set_title(f'T={i}'); axes[i].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            preview_path = "images/dream_sequence_preview.png"
            plt.savefig(preview_path); print(f"Saved preview plot to {preview_path}"); plt.close(fig)
        except Exception as e: print(f"Error saving preview plot: {e}")