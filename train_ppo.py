# train_ppo.py
import time
import argparse

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


def get_config(name="default"):
    configs = {
        "default": {
            "GAMMA": 0.99,
            "LAMBDA": 0.95,
            "EPSILON": 0.2,
            "ACTOR_LR": 1e-4,
            "CRITIC_LR": 3e-4,
            "EPOCHS_PER_UPDATE": 10,
            "MINIBATCH_SIZE": 64,
            "STEPS_PER_BATCH": 2048,
            "MAX_TRAINING_STEPS": 10_000_000,
            "INITIAL_ENTROPY_COEF": 0.01,
            "FINAL_ENTROPY_COEF": 0.001,
            "ENTROPY_ANNEAL_FRACTION": 0.75,
            "VF_COEF": 0.5,
            "TARGET_KL": 0.015,
            "GRAD_CLIP_NORM": 0.5,
            "SAVE_INTERVAL": 50,
        }
    }
    # copy default and update for test
    configs["test"] = configs["default"].copy()
    configs["test"]["MAX_TRAINING_STEPS"] = 1000
    configs["test"]["STEPS_PER_BATCH"] = 128
    configs["test"]["INITIAL_ENTROPY_COEF"] = 0.01
    configs["test"]["FINAL_ENTROPY_COEF"] = 0.001
    configs["test"]["ENTROPY_ANNEAL_FRACTION"] = 0.75


    return configs[name]


# --- Main Training Function ---
def train_ppo(config):  # Added config argument
    print("Starting PPO training...")
    start_time = time.time()

    # Put hyperparams into object
    hyperparams = PPOHyperparameters(
        gamma=config["GAMMA"], lambda_gae=config["LAMBDA"], epsilon_clip=config["EPSILON"],
        actor_lr=config["ACTOR_LR"], critic_lr=config["CRITIC_LR"], epochs_per_update=config["EPOCHS_PER_UPDATE"],
        minibatch_size=config["MINIBATCH_SIZE"], vf_coef=config["VF_COEF"],
        grad_clip_norm=config["GRAD_CLIP_NORM"], target_kl=config["TARGET_KL"]
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
        gamma=config["GAMMA"],
    )

    # 3. Initialize Actor, Critic, Optimizers
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config["ACTOR_LR"], eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config["CRITIC_LR"], eps=1e-5)

    # Store initial learning rates for annealing
    initial_actor_lr = config["ACTOR_LR"]
    initial_critic_lr = config["CRITIC_LR"]

    # 4. Initialize Rollout Buffer
    buffer = RolloutBuffer()

    # 5. Training Loop
    global_step = 0
    update = 0
    all_episode_rewards = []
    current_Z_t_numpy, _ = env.reset()  # (num_stack * LATENT_DIM,)

    while global_step < config["MAX_TRAINING_STEPS"]:
        update += 1

        # --- Learning Rate Annealing ---
        progress_fraction = global_step / config["MAX_TRAINING_STEPS"]
        decayed_actor_lr = initial_actor_lr * (1.0 - progress_fraction)
        decayed_critic_lr = initial_critic_lr * (1.0 - progress_fraction)

        # Update optimizer learning rates
        actor_optimizer.param_groups[0]['lr'] = decayed_actor_lr
        critic_optimizer.param_groups[0]['lr'] = decayed_critic_lr

        # --- Entropy Coefficient Annealing ---
        if config["ENTROPY_ANNEAL_FRACTION"] > 0:
            entropy_anneal_progress = min(1.0, progress_fraction / config["ENTROPY_ANNEAL_FRACTION"])
        else: # Avoid division by zero, default to initial if fraction is 0 or less
            entropy_anneal_progress = 0.0
        current_entropy_coef = config["INITIAL_ENTROPY_COEF"] * (1.0 - entropy_anneal_progress) + \
                               config["FINAL_ENTROPY_COEF"] * entropy_anneal_progress

        buffer.clear()
        actor.eval()  # Set to eval mode for rollout collection
        critic.eval()
        current_episode_reward = 0
        # num_steps_this_batch = 0

        # --- Collect Rollout Data (STEPS_PER_BATCH) ---
        for _ in range(config["STEPS_PER_BATCH"]):
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
            if global_step >= config["MAX_TRAINING_STEPS"]:
                break

        # --- Prepare for Update ---
        # Compute value for the last state reached
        with torch.no_grad():
            Z_last_tensor = torch.tensor(current_Z_t_numpy, dtype=torch.float32).to(DEVICE)
            last_value = critic(Z_last_tensor.unsqueeze(0)).squeeze().to(DEVICE)

        # --- Perform PPO Update ---
        mean_kl = perform_ppo_update(
            buffer, actor, critic, actor_optimizer, critic_optimizer,
            hyperparams, DEVICE, last_value_for_gae=last_value,
            current_entropy_coef=current_entropy_coef  # Pass annealed entropy coef
        )

        print(f"Update {update}, Optimizing for {config['EPOCHS_PER_UPDATE']} epochs. Mean KL: {mean_kl:.4f}, Current LRs (A/C): {decayed_actor_lr:.2e}/{decayed_critic_lr:.2e}, Entropy Coef: {current_entropy_coef:.2e}")
        avg_reward = np.mean(all_episode_rewards[-10:]) if all_episode_rewards else 0.0
        print(f"Total Steps: {global_step}, Avg Reward (Last 10 ep): {avg_reward:.2f}")

        # --- Save Models Periodically ---
        if update % config["SAVE_INTERVAL"] == 0:
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

    parser = argparse.ArgumentParser(description="Train PPO agent.")
    parser.add_argument(
        "--config_name",
        type=str,
        default="default",
        help="Name of the configuration to use (e.g., 'default', 'test')."
    )
    args = parser.parse_args()

    print(f"Using configuration name: {args.config_name}")
    config = get_config(args.config_name)

    train_ppo(config)
