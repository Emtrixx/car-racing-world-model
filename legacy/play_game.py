# play_game.py
import gymnasium as gym
import torch
import numpy as np
import time

# Import from local modules
from utils import (DEVICE, ENV_NAME, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode_stack, FrameStackWrapper)
# Assuming PPO filenames were also added to utils, or define them here
# PPO_ACTOR_SAVE_FILENAME = f"{ENV_NAME}_ppo_actor_ld{LATENT_DIM}.pth" # Define if not in utils
from legacy.utils_rl import PPO_ACTOR_SAVE_FILENAME

from actor_critic import Actor
from legacy.conv_vae import ConvVAE

# --- Configuration ---
NUM_EPISODES = 5  # How many episodes to play
PLAYBACK_SPEED_DELAY = 0.02  # Seconds to pause between steps (increase for slower playback)


def play():
    print(f"Initializing environment: {ENV_NAME} with human rendering.")
    # Use render_mode="human"
    env = FrameStackWrapper(gym.make(ENV_NAME, render_mode="human", max_episode_steps=1000))

    # --- Load Models ---
    print(f"Loading models to device: {DEVICE}")

    # Load VAE
    vae_model = ConvVAE().to(DEVICE)  # Assumes defaults match trained model
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found.")
        env.close();
        return
    except Exception as e:
        print(f"ERROR loading VAE: {e}");
        env.close();
        return

    # Load Actor (Policy)
    actor_model = Actor().to(DEVICE)  # Assumes defaults match trained model
    try:
        actor_model.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
        actor_model.eval()
        print(f"Successfully loaded Actor: {PPO_ACTOR_SAVE_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: Actor checkpoint '{PPO_ACTOR_SAVE_FILENAME}' not found.")
        env.close();
        return
    except Exception as e:
        print(f"ERROR loading Actor: {e}");
        env.close();
        return

    # --- Play Episodes ---
    all_rewards = []
    for episode in range(NUM_EPISODES):
        print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES}")
        obs, info = env.reset()
        done = False
        truncated = False  # Gymnasium uses truncated separately
        total_reward = 0
        step_count = 0

        while not done and not truncated:
            # 1. Preprocess and Encode Observation
            with torch.no_grad():
                # preprocess and encode stack
                z_stack_t = preprocess_and_encode_stack(obs, transform, vae_model, DEVICE)

                # 2. Get Action from Policy (Actor)
                dist = actor_model(z_stack_t.unsqueeze(0))  # Add batch dimension

                # Choose deterministic action for evaluation/playback (greedy)
                action_raw = dist.mean  # Use mean for deterministic action
                # action_raw = dist.sample() # random sample

                # 3. Process Action (tanh squashing, scaling, clipping) - same as in training
                action_processed = torch.tanh(action_raw)  # Already applied tanh to steering mean in model
                # Scale/shift gas/brake if necessary (applied to raw action from sample/mean)
                # Assuming action dims are [Steering, Gas, Brake]
                action_processed = torch.cat([
                    action_processed[:, :1],  # Steering already in [-1, 1] approx
                    (action_processed[:, 1:] + 1.0) / 2.0  # Gas, Brake -> [0, 1] approx
                ], dim=1)

                # Clip to ensure bounds
                env_low = torch.tensor(env.action_space.low, device=DEVICE, dtype=torch.float32)
                env_high = torch.tensor(env.action_space.high, device=DEVICE, dtype=torch.float32)
                action_clipped = torch.clamp(action_processed, env_low, env_high)

                # Convert to numpy for environment
                action_np = action_clipped.squeeze(0).cpu().numpy()  # Remove batch dim

            # 4. Step Environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated  # Check both flags

            total_reward += reward
            step_count += 1

            # Optional delay for visibility
            if PLAYBACK_SPEED_DELAY > 0:
                time.sleep(PLAYBACK_SPEED_DELAY)

            # Optional: Render explicitly if needed, though "human" mode usually handles it
            # env.render()

        print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    # --- Cleanup ---
    env.close()
    print("\nFinished playing.")
    if all_rewards:
        print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    play()
