# train_ppo_in_dream.py
from collections import deque

import gymnasium as gym  # For action_space if needed for RandomPolicy fallback
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F  # For PPO loss, e.g. F.mse_loss
import time
import matplotlib.pyplot as plt

from torch import nn, random

from legacy.actor_critic import Actor, Critic
from legacy.conv_vae import ConvVAE
from world_model import WorldModelGRU
# Import from local modules
from utils import (DEVICE, ENV_NAME, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode)
from legacy.utils_rl import RandomPolicy, PPOHyperparameters, PPO_DREAM_ACTOR_SAVE_FILENAME, \
    PPO_DREAM_CRITIC_SAVE_FILENAME, RolloutBuffer

# Import GRU WM parameters and checkpoint path from its training script
# This assumes train_world_model.py defines these at the global scope
from legacy.train_world_model import (WM_CHECKPOINT_FILENAME_GRU, GRU_HIDDEN_DIM,
                                      GRU_NUM_LAYERS, GRU_INPUT_EMBED_DIM)

print(f"Device for PPO in Dream: {DEVICE}")

# --- Dream-Specific Hyperparameters ---
DREAM_STEPS_PER_BATCH = 2048  # How many steps to dream per PPO update
MAX_DREAM_TRAINING_UPDATES = 500  # Number of PPO updates using dreamed data
DREAM_HORIZON_PER_EPISODE = 200  # Max length of a single dream sequence before reset
INITIAL_STATES_BUFFER_SIZE = 10000  # Size of buffer for real initial states for dreaming
INITIAL_STATES_COLLECT_STEPS = 2000  # Steps with random policy to populate initial states buffer
DONE_THRESHOLD = 0.5  # Threshold for sigmoid(done_logit) to consider an episode done in dream

DREAM_SAVE_INTERVAL = 20


# --- Helper to collect initial states from real environment ---
def collect_initial_real_states(env, vae_model, transform_fn, num_steps, buffer_size, device):
    print(f"Collecting {num_steps} real steps for initial dream states...")
    real_z_buffer = deque(maxlen=buffer_size)
    obs, _ = env.reset()
    temp_policy = RandomPolicy(env.action_space)  # Use random policy for initial state collection

    for _ in range(num_steps):
        z_t = preprocess_and_encode(obs, transform_fn, vae_model, device)
        real_z_buffer.append(z_t.cpu())  # Store on CPU

        action = temp_policy.get_action(obs)  # Action doesn't matter much, just need states
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"Collected {len(real_z_buffer)} initial latent states.")
    return list(real_z_buffer)


# --- PPO Update Function (Refactored from train_ppo.py for reusability) ---
# This function would ideally be in a shared utils_rl.py or similar
def perform_ppo_update(buffer, actor, critic, actor_optimizer, critic_optimizer,
                       epochs, minibatch_size, gamma, lambda_, epsilon_clip,
                       vf_coef, entropy_coef, grad_clip_norm, target_kl, device):
    actor.train()
    critic.train()
    approx_kl_divs_epoch = []

    # Compute last value for GAE (not strictly needed if dreams end with done, but good practice)
    # For dreaming, if a dream ends because horizon is met, last_value could be V(z_final_dream_state)
    # For simplicity, if all dreams end due to predicted done or horizon, last_value can be 0.
    # Here, we assume buffer.values contains V(s_t) and last_value is handled if a dream doesn't naturally end.
    # This part needs careful thought for dream scenarios.
    # For now, let's assume the buffer's last entry has a value and done flag that GAE can use.
    # If a dream ends prematurely (not by predicted done), the value of the last state needs to be estimated.
    # For simplicity, we will assume dream rollouts naturally "end" or the final value is 0 if horizon is met.
    if len(buffer.values) > 0:
        last_val_for_gae = buffer.values[-1] if not buffer.dones[-1] else torch.tensor(0.0,
                                                                                       device=buffer.values[-1].device)
    else:  # Buffer might be empty if dream_steps_per_batch is small
        return 0.0  # No update if buffer is empty

    buffer.compute_returns_and_advantages(
        last_value=last_val_for_gae,  # This needs careful handling for dreams
        gamma=gamma, lambda_=lambda_
    )
    buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    for epoch in range(epochs):
        approx_kl_divs_minibatch = []
        for states, actions, old_log_probs, returns, advantages in buffer.get_batch(minibatch_size):
            states, actions, old_log_probs, returns, advantages = \
                states.to(device), actions.to(device), old_log_probs.to(device), \
                    returns.to(device), advantages.to(device)

            actions, old_log_probs, returns, advantages = \
                actions.detach(), old_log_probs.detach(), returns.detach(), advantages.detach()

            new_dist = actor(states)
            new_log_probs = new_dist.log_prob(actions).sum(1)
            entropy = new_dist.entropy().sum(1).mean()
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            new_values = critic(states).squeeze(1)
            critic_loss = F.mse_loss(new_values, returns)
            loss = actor_loss + vf_coef * critic_loss - entropy_coef * entropy

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
            actor_optimizer.step()
            nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
            critic_optimizer.step()

            with torch.no_grad():
                kl = (ratio - 1) - log_ratio
                approx_kl_divs_minibatch.append(kl.mean().item())

        epoch_kl_mean = np.mean(approx_kl_divs_minibatch)
        approx_kl_divs_epoch.append(epoch_kl_mean)
        if target_kl is not None and epoch_kl_mean > target_kl:
            print(f"  Epoch {epoch + 1}: Early stopping at KL divergence {epoch_kl_mean:.4f} > {target_kl}")
            break
    return np.mean(approx_kl_divs_epoch)


# --- Main Training in Dream ---
def train_ppo_in_dream():
    print("Starting PPO training in Dream Environment...")
    run_start_time = time.time()

    # 1. Load VAE
    vae_model = ConvVAE().to(DEVICE);
    vae_model.eval()
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load VAE: {e}");
        return
    print("VAE loaded.")

    # 2. Load World Model (GRU with R/D heads)
    world_model = WorldModelGRU(
        gru_hidden_dim=GRU_HIDDEN_DIM, gru_num_layers=GRU_NUM_LAYERS,
        gru_input_embed_dim=GRU_INPUT_EMBED_DIM
    ).to(DEVICE);
    world_model.eval()
    try:
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load GRU World Model: {e}");
        return
    print("GRU World Model loaded.")

    # 3. Initialize or Load PPO Actor/Critic
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    # Optional: Load pre-trained PPO models from real env training
    # try: actor.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
    # except: print("Initializing new PPO Actor for dream.")
    # try: critic.load_state_dict(torch.load(PPO_CRITIC_SAVE_FILENAME, map_location=DEVICE))
    # except: print("Initializing new PPO Critic for dream.")

    # Instantiate PPOHyperparameters todo: use a config file or shared module
    hyperparams = PPOHyperparameters(
        gamma=0.99, lambda_gae=0.95, epsilon_clip=0.2,
        actor_lr=3e-4, critic_lr=1e-3, epochs_per_update=4,  # Fewer epochs often better for model-based
        minibatch_size=64, entropy_coef=0.01, vf_coef=0.5,
        grad_clip_norm=0.5, target_kl=0.015
    )

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-5)
    print("PPO Actor/Critic initialized.")

    # 4. Populate initial states buffer (optional, but recommended for grounding)
    temp_env = gym.make(ENV_NAME, render_mode="rgb_array")  # for initial states
    initial_z_states = collect_initial_real_states(temp_env, vae_model, transform,
                                                   INITIAL_STATES_COLLECT_STEPS,
                                                   INITIAL_STATES_BUFFER_SIZE, DEVICE)
    temp_env.close()
    if not initial_z_states:
        print("No initial states collected, cannot start dreaming.")
        return

    # 5. PPO Training Loop in Dream
    dream_buffer = RolloutBuffer()
    all_dream_episode_rewards = []  # Track rewards *within* dreams

    for update_num in range(1, MAX_DREAM_TRAINING_UPDATES + 1):
        dream_buffer.clear()
        actor.eval();
        critic.eval()  # For collecting dream data

        # Select a starting state for the dream batch
        current_z = random.choice(initial_z_states).to(DEVICE)
        current_h = torch.zeros(GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM).to(DEVICE)  # Batch size 1

        total_dream_reward_this_batch = 0
        num_dream_episodes_this_batch = 0

        for dream_step in range(DREAM_STEPS_PER_BATCH):
            # Get action from current PPO policy
            with torch.no_grad():
                value_z = critic(current_z.unsqueeze(0)).squeeze()  # V(z_t)
                dist = actor(current_z.unsqueeze(0))
                action_raw = dist.sample()
                log_prob_action = dist.log_prob(action_raw).sum(1).squeeze(0)

                # Process action (tanh, scale, clip) - similar to PPOPolicyWrapper
                action_processed = torch.tanh(action_raw.squeeze(0))  # action_raw is (1, act_dim)
                action_scaled = torch.zeros_like(action_processed)
                action_scaled[0] = action_processed[0]  # Steering
                action_scaled[1:] = (action_processed[1:] + 1.0) / 2.0  # Gas, Brake

                # Use fixed action space bounds for CarRacing-v3
                env_low = torch.tensor([-1.0, 0.0, 0.0], device=DEVICE, dtype=torch.float32)
                env_high = torch.tensor([1.0, 1.0, 1.0], device=DEVICE, dtype=torch.float32)
                action_clipped = torch.clamp(action_scaled, env_low, env_high)

            # Dream one step using World Model
            with torch.no_grad():
                next_z_pred, reward_pred, done_logit_pred, next_h = \
                    world_model.step(current_z.unsqueeze(0), action_clipped.unsqueeze(0), current_h)
                # Unsqueeze for batch dim 1, then squeeze results
                next_z_pred = next_z_pred.squeeze(0)
                reward_pred = reward_pred.squeeze(0)  # Scalar tensor
                done_pred_prob = torch.sigmoid(done_logit_pred.squeeze(0))  # Scalar tensor
                dream_done = (done_pred_prob > DONE_THRESHOLD).float()  # Convert to 0.0 or 1.0

            total_dream_reward_this_batch += reward_pred.item()

            dream_buffer.add(current_z.cpu(), action_raw.squeeze(0).cpu(), log_prob_action.cpu(),
                             reward_pred.cpu(), dream_done.cpu(), value_z.cpu())

            current_z = next_z_pred
            current_h = next_h

            if dream_done.item() > 0.5 or (dream_step + 1) % DREAM_HORIZON_PER_EPISODE == 0:
                all_dream_episode_rewards.append(
                    total_dream_reward_this_batch / (num_dream_episodes_this_batch + 1e-6))  # Avg reward per episode
                num_dream_episodes_this_batch += 1
                # Reset for next dream episode
                current_z = random.choice(initial_z_states).to(DEVICE)
                current_h = torch.zeros(GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM).to(DEVICE)
                if num_dream_episodes_this_batch > 0: total_dream_reward_this_batch = 0  # Reset counter if we had episodes

                # --- Determine last_value_for_gae for dream ---
                with torch.no_grad():
                    if dream_buffer.dones and dream_buffer.dones[-1].item() > 0.5:  # If last dream step was 'done'
                        last_value_for_gae_dream = torch.tensor(0.0, device=DEVICE)
                    elif dream_buffer.states:  # If buffer is not empty
                        # Use critic to estimate value of the very last dreamed state
                        last_dreamed_z = dream_buffer.states[-1].to(DEVICE)  # Ensure it's the state *after* last action
                        # This needs to be the z_next from the final step of collection
                        # The GAE calculation expects V(s_T+1).
                        # So, if current_z is the last state added to buffer.states,
                        # last_value_for_gae should be V(current_z_after_last_dream_step)
                        # If a dream ends by horizon, current_z would be the state to evaluate.
                        # For simplicity, let's get z_T (last state in dream_buffer.states) and its value.
                        # The logic within perform_ppo_update.buffer.compute_returns_and_advantages handles
                        # whether this last_value is used based on the last done flag.
                        # For the last step in the dream batch current_z is z_{T_batch}
                        # We need V(z_{T_batch}) if not done, or V(z_{T_batch+1}) if it's the value after the last action.
                        # The `current_z` at the end of the dream collection loop is the state *after* the last action.
                        last_value_for_gae_dream = critic(current_z.unsqueeze(0)).squeeze(0).to(DEVICE)
                    else:  # Buffer empty
                        last_value_for_gae_dream = torch.tensor(0.0, device=DEVICE)

        # Perform PPO update using the dream_buffer
        mean_kl = perform_ppo_update(
            dream_buffer, actor, critic, actor_optimizer, critic_optimizer,
            hyperparams, DEVICE, last_value_for_gae_dream
        )
        avg_dream_ep_reward = np.mean(all_dream_episode_rewards[-10:]) if all_dream_episode_rewards else 0
        print(f"Dream Update: {update_num}, Avg KL: {mean_kl:.4f}, Avg Dream Ep Reward: {avg_dream_ep_reward:.2f}")

        if update_num % DREAM_SAVE_INTERVAL == 0:
            print("Saving PPO models trained in dream...")
            torch.save(actor.state_dict(), PPO_DREAM_ACTOR_SAVE_FILENAME)
            torch.save(critic.state_dict(), PPO_DREAM_CRITIC_SAVE_FILENAME)

    print(f"Finished PPO training in dream. Total time: {(time.time() - run_start_time):.2f}s")

    print("Saving PPO models trained in dream...")
    torch.save(actor.state_dict(), PPO_DREAM_ACTOR_SAVE_FILENAME)
    torch.save(critic.state_dict(), PPO_DREAM_CRITIC_SAVE_FILENAME)

    # Plot dream rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_dream_episode_rewards)
    plt.xlabel("Dream Episode Index (approx)")
    plt.ylabel("Total Predicted Reward in Dream")
    plt.title("PPO Training Rewards in Dream Environment")
    plt.savefig("images/ppo_dream_training_rewards.png")
    print("Saved dream rewards plot.")


if __name__ == "__main__":
    # Path(PPO_DREAM_ACTOR_SAVE_FILENAME).parent.mkdir(parents=True, exist_ok=True) # If saving in subdir
    train_ppo_in_dream()
