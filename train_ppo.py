# train_ppo.py
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from actor_critic import Actor, Critic
from conv_vae import ConvVAE
# Import from local modules
from utils import (DEVICE, ENV_NAME, transform,
                   VAE_CHECKPOINT_FILENAME, NUM_STACK,
                   make_env, LATENT_DIM)
from utils_rl import perform_ppo_update, PPO_ACTOR_SAVE_FILENAME, \
    PPO_CRITIC_SAVE_FILENAME, PPOHyperparameters, RolloutBuffer

print(f"Using device: {DEVICE}")

# --- PPO Hyperparameters ---
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # Lambda for GAE
EPSILON = 0.2  # Clipping parameter for PPO
ACTOR_LR = 1e-4  # Learning rate for actor
CRITIC_LR = 3e-4  # Learning rate for critic
# EPOCHS_PER_UPDATE = 5  # testing
EPOCHS_PER_UPDATE = 10  # Number of optimization epochs per batch
MINIBATCH_SIZE = 64
STEPS_PER_BATCH = 2048  # Number of steps to collect rollout data per update
# MAX_TRAINING_STEPS = 10_000  # testing
MAX_TRAINING_STEPS = 10_000_000  # Total steps for training
ENTROPY_COEF = 0.01  # Entropy regularization coefficient
VF_COEF = 0.5  # Value function loss coefficient
TARGET_KL = 0.015  # Target KL divergence limit (optional, for early stopping updates)
GRAD_CLIP_NORM = 0.5  # Gradient clipping norm

# --- Saving ---
SAVE_INTERVAL = 50  # Save models every N updates


# --- Main Training Function ---
def train_ppo():
    print("Starting PPO training...")
    start_time = time.time()

    # Put hyperparams into object
    hyperparams = PPOHyperparameters(
        gamma=GAMMA, lambda_gae=LAMBDA, epsilon_clip=EPSILON,
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, epochs_per_update=EPOCHS_PER_UPDATE,
        minibatch_size=MINIBATCH_SIZE, entropy_coef=ENTROPY_COEF, vf_coef=VF_COEF,
        grad_clip_norm=GRAD_CLIP_NORM, target_kl=TARGET_KL
    )

    # Load ConvVAE
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        return
    except Exception as e:
        print(f"ERROR loading VAE: {e}");
        return

    # Setup Environment
    # - wrapper pipeline
    # - raw observations are preprocessed and embedded
    # - actions and rewards are clipped
    env = make_env(
        env_id=ENV_NAME,
        vae_model_instance=vae_model,
        device_for_vae=DEVICE,
        frame_stack_num=NUM_STACK,
        transform_function=transform,
        single_latent_dim=LATENT_DIM,
        gamma=GAMMA,
    )

    # 3. Initialize Actor, Critic, Optimizers
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-5)

    # 4. Initialize Rollout Buffer
    buffer = RolloutBuffer()

    # 5. Training Loop
    global_step = 0
    update = 0
    all_episode_rewards = []
    current_Z_t_numpy, _ = env.reset()  # (num_stack * LATENT_DIM,)

    while global_step < MAX_TRAINING_STEPS:
        update += 1
        buffer.clear()
        actor.eval()  # Set to eval mode for rollout collection
        critic.eval()
        current_episode_reward = 0
        # num_steps_this_batch = 0

        # --- Collect Rollout Data (STEPS_PER_BATCH) ---
        for _ in range(STEPS_PER_BATCH):
            global_step += 1
            # num_steps_this_batch += 1

            # Convert current NumPy state from env to PyTorch tensor for actor/critic
            Z_t_tensor = torch.tensor(current_Z_t_numpy, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                value = critic(Z_t_tensor.unsqueeze(0)).squeeze()

            # Sample action from actor policy
            dist = actor(Z_t_tensor.unsqueeze(0))
            action_raw_from_dist = dist.sample()  # This is the 'raw' action from distribution
            log_prob = dist.log_prob(action_raw_from_dist).sum(1).squeeze(0)
            action_raw_from_dist = action_raw_from_dist.squeeze(0)

            # --- Action Clipping/Processing ---
            # Apply tanh squashing *after* sampling (more stable than squashing mean)
            # action dims are [Steering, Gas, Brake]
            # For actions in [-1, 1] (Steering)
            action_processed = torch.tanh(action_raw_from_dist)
            # For actions in [0, 1] (Gas, Brake) - shift and scale tanh output
            action_processed = torch.cat([
                action_processed[:1],  # Steering already in [-1, 1]
                (action_processed[1:] + 1.0) / 2.0  # Gas, Brake -> [0, 1]
            ], dim=0)
            # Clip to ensure bounds (important!)
            action_clipped = torch.clamp(action_processed,
                                         torch.tensor(env.action_space.low, device=DEVICE),
                                         torch.tensor(env.action_space.high, device=DEVICE))
            action_np = action_clipped.detach().cpu().numpy()

            # Step the environment
            next_Z_t_numpy, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            current_episode_reward += reward  # Use reward from NormalizeReward wrapper

            # Store transition data (move tensors to CPU for storage). State is Z_t, action is raw from distribution
            buffer.add(Z_t_tensor.cpu(), action_raw_from_dist.cpu(), log_prob.cpu(),
                       torch.tensor(reward, dtype=torch.float32).cpu(),
                       torch.tensor(done, dtype=torch.bool).cpu(),
                       value.cpu())

            current_Z_t_numpy = next_Z_t_numpy  # Update observation for next step
            if done:
                print(f"Step: {global_step}, Episode Reward: {current_episode_reward:.2f}")
                all_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                current_Z_t_numpy, _ = env.reset()

            # Check if max steps reached during collection
            if global_step >= MAX_TRAINING_STEPS:
                break

        # --- Prepare for Update ---
        # Compute value for the last state reached
        with torch.no_grad():
            Z_last_tensor = torch.tensor(current_Z_t_numpy, dtype=torch.float32).to(DEVICE)
            last_value = critic(Z_last_tensor.unsqueeze(0)).squeeze().to(DEVICE)

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
