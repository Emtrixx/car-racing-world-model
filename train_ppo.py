# train_ppo.py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from actor_critic import Actor, Critic
from utils_rl import perform_ppo_update, PPO_ACTOR_SAVE_FILENAME, \
    PPO_CRITIC_SAVE_FILENAME, PPOHyperparameters, RolloutBuffer
# Import from local modules
from utils import (DEVICE, ENV_NAME, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode)
from conv_vae import ConvVAE

print(f"Using device: {DEVICE}")

# --- PPO Hyperparameters ---
GAMMA = 0.99           # Discount factor
LAMBDA = 0.95          # Lambda for GAE
EPSILON = 0.2          # Clipping parameter for PPO
ACTOR_LR = 1e-4        # Learning rate for actor
CRITIC_LR = 3e-4       # Learning rate for critic
EPOCHS_PER_UPDATE = 10 # Number of optimization epochs per batch
MINIBATCH_SIZE = 64
STEPS_PER_BATCH = 2048 # Number of steps to collect rollout data per update
MAX_TRAINING_STEPS = 500_000 # Total steps for training todo: higher
ENTROPY_COEF = 0.01    # Entropy regularization coefficient
VF_COEF = 0.5          # Value function loss coefficient
TARGET_KL = 0.015      # Target KL divergence limit (optional, for early stopping updates)
GRAD_CLIP_NORM = 0.5   # Gradient clipping norm

# --- Saving ---
SAVE_INTERVAL = 50 # Save models every N updates


# --- Main Training Function ---
def train_ppo():
    print("Starting PPO training...")
    start_time = time.time()

    # 1. Setup Environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    # Note: For continuous actions, CarRacing-v3 might benefit from wrappers like:
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -1.0, 1.0)) # Reward clipping
    # env = gym.wrappers.NormalizeObservation(env) # If not using VAE
    env = gym.wrappers.NormalizeReward(env, gamma=GAMMA) # If needed

    # 2. Load VAE (ensure it's trained)
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        env.close(); return
    except Exception as e:
        print(f"ERROR loading VAE: {e}"); env.close(); return

    # 3. Initialize Actor, Critic, Optimizers
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-5)

    hyperparams = PPOHyperparameters(
        gamma=GAMMA, lambda_gae=LAMBDA, epsilon_clip=EPSILON,
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, epochs_per_update=EPOCHS_PER_UPDATE,
        minibatch_size=MINIBATCH_SIZE, entropy_coef=ENTROPY_COEF, vf_coef=VF_COEF,
        grad_clip_norm=GRAD_CLIP_NORM, target_kl=TARGET_KL
    )

    # 4. Initialize Rollout Buffer
    buffer = RolloutBuffer()

    # 5. Training Loop
    global_step = 0
    update = 0
    all_episode_rewards = []
    start_obs, _ = env.reset()

    while global_step < MAX_TRAINING_STEPS:
        update += 1
        buffer.clear()
        actor.eval() # Set to eval mode for rollout collection
        critic.eval()
        current_episode_reward = 0
        num_steps_this_batch = 0

        # --- Collect Rollout Data (STEPS_PER_BATCH) ---
        for _ in range(STEPS_PER_BATCH):
            global_step += 1
            num_steps_this_batch += 1

            # Encode observation to latent state z_t
            with torch.no_grad():
                z_t = preprocess_and_encode(start_obs, transform, vae_model, DEVICE)
                value = critic(z_t.unsqueeze(0)).squeeze(0) # Add/remove batch dim

            # Sample action from actor policy
            dist = actor(z_t.unsqueeze(0)) # Add batch dim
            action = dist.sample() # Sample raw action
            log_prob = dist.log_prob(action).sum(1) # Sum log_prob across action dims
            action = action.squeeze(0) # Remove batch dim
            log_prob = log_prob.squeeze(0)

            # --- Action Clipping/Processing ---
            # Apply tanh squashing *after* sampling (more stable than squashing mean)
            # For actions in [-1, 1] (Steering)
            action_processed = torch.tanh(action)
            # For actions in [0, 1] (Gas, Brake) - shift and scale tanh output
            # Assuming action dims are [Steering, Gas, Brake]
            action_processed = torch.cat([
                 action_processed[:1], # Steering already in [-1, 1]
                 (action_processed[1:] + 1.0) / 2.0 # Gas, Brake -> [0, 1]
            ], dim=0)
            # Clip to ensure bounds (important!)
            action_clipped = torch.clamp(action_processed,
                                         torch.tensor(env.action_space.low, device=DEVICE),
                                         torch.tensor(env.action_space.high, device=DEVICE))
            action_np = action_clipped.detach().cpu().numpy()


            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            current_episode_reward += reward

            # Store transition data (move tensors to CPU for storage)
            buffer.add(z_t.cpu(), action.cpu(), log_prob.cpu(),
                       torch.tensor(reward, dtype=torch.float32).cpu(),
                       torch.tensor(done, dtype=torch.bool).cpu(),
                       value.cpu())

            start_obs = next_obs # Update observation
            if done:
                print(f"Step: {global_step}, Episode Reward: {current_episode_reward:.2f}")
                all_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                start_obs, _ = env.reset()

            # Check if max steps reached during collection
            if global_step >= MAX_TRAINING_STEPS:
                break

        # --- Prepare for Update ---
        # Compute value for the last state reached
        with torch.no_grad():
            z_last = preprocess_and_encode(start_obs, transform, vae_model, DEVICE)
            last_value = critic(z_last.unsqueeze(0)).squeeze(0)
        # Compute returns and advantages using GAE
        buffer.returns, buffer.advantages = buffer.compute_returns_and_advantages(last_value.cpu(), GAMMA, LAMBDA)

        # Normalize advantages (important for stability)
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        # --- Perform PPO Update ---
        mean_kl = perform_ppo_update(
            buffer, actor, critic, actor_optimizer, critic_optimizer,
            hyperparams, DEVICE, last_value_for_gae=last_value
        )


        print(f"Update {update}, Optimizing for {EPOCHS_PER_UPDATE} epochs. Mean KL: {mean_kl:.4f}")
        avg_reward = np.mean(all_episode_rewards[-10:]) if all_episode_rewards else 0.0
        print(f"Total Steps: {global_step}, Avg Reward (Last 10 ep): {avg_reward:.2f}")


        # --- Save Models Periodically ---
        if update % SAVE_INTERVAL == 0:
            print(f"Saving models at update {update}...")
            try:
                torch.save(actor.state_dict(), PPO_ACTOR_SAVE_FILENAME)
                torch.save(critic.state_dict(), PPO_CRITIC_SAVE_FILENAME)
                print("Models saved successfully.")
            except Exception as e:
                print(f"Error saving models: {e}")

    # --- End of Training ---
    try:
        torch.save(actor.state_dict(), PPO_ACTOR_SAVE_FILENAME)
        torch.save(critic.state_dict(), PPO_CRITIC_SAVE_FILENAME)
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")

    env.close()
    print("Training finished.")
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Optional: Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Rewards")
    plt.savefig("images/ppo_training_rewards.png")
    print("Saved rewards plot to ppo_training_rewards.png")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure save directories exist if paths include directories
    # Path(PPO_ACTOR_SAVE_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    # Path(PPO_CRITIC_SAVE_FILENAME).parent.mkdir(parents=True, exist_ok=True)

    train_ppo()