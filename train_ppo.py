# train_ppo.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal # Using Gaussian policy for continuous actions
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Import from local modules
from utils import (DEVICE, ENV_NAME, LATENT_DIM, ACTION_DIM, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode)
from models import ConvVAE

print(f"Using device: {DEVICE}")

# --- PPO Hyperparameters ---
GAMMA = 0.99           # Discount factor
LAMBDA = 0.95          # Lambda for GAE
EPSILON = 0.2          # Clipping parameter for PPO
ACTOR_LR = 3e-4        # Learning rate for actor
CRITIC_LR = 1e-3       # Learning rate for critic
EPOCHS_PER_UPDATE = 5 # Number of optimization epochs per batch
MINIBATCH_SIZE = 64
STEPS_PER_BATCH = 2048 # Number of steps to collect rollout data per update
MAX_TRAINING_STEPS = 100_000 # Total steps for training todo: higher
ENTROPY_COEF = 0.01    # Entropy regularization coefficient
VF_COEF = 0.5          # Value function loss coefficient
TARGET_KL = 0.015      # Target KL divergence limit (optional, for early stopping updates)
GRAD_CLIP_NORM = 0.5   # Gradient clipping norm

# --- File Paths & Saving ---
PPO_ACTOR_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_actor_ld{LATENT_DIM}.pth"
PPO_CRITIC_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_critic_ld{LATENT_DIM}.pth"
SAVE_INTERVAL = 50 # Save models every N updates

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Output layer for action means
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # Output layer for action log standard deviations (log_std)
        # Using a learnable parameter per action dimension, not state-dependent initially
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        action_mean = self.fc_mean(x)

        # We use tanh activation on the mean for steering [-1, 1].
        # For gas/brake [0, 1], we could apply sigmoid or (tanh+1)/2 later,
        # but often letting the distribution + clipping handle it works okay.
        # Let's apply tanh to the first dim (steering) explicitly.
        # Keep gas/brake means unbounded for now, will rely on sampling/clipping.
        action_mean = torch.cat([
            torch.tanh(action_mean[:, :1]), # Steering mean bounded [-1, 1]
            action_mean[:, 1:]              # Gas, Brake means unbounded
        ], dim=1)


        action_log_std = self.log_std.expand_as(action_mean) # Same log_std for all states
        action_std = torch.exp(action_log_std)

        # Create the Normal distribution
        dist = Normal(action_mean, action_std)
        return dist

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output a single value
        )

    def forward(self, state):
        return self.net(state)

# --- PPO Storage ---
# Simple class or dictionary to hold rollout data
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma, lambda_):
        """ Computes GAE and returns. """
        n_steps = len(self.rewards)
        if n_steps == 0:
            self.returns = torch.tensor([])
            self.advantages = torch.tensor([])
            return [], []  # Handle empty buffer case

        # Ensure tensors created on CPU since buffer stores CPU tensors
        advantages = torch.zeros(n_steps, dtype=torch.float32, device='cpu')
        returns = torch.zeros(n_steps, dtype=torch.float32, device='cpu')

        # Stack values for easier access and ensure it's on CPU
        cpu_values = torch.stack(self.values)  # self.values are stored on CPU
        last_value_cpu = last_value.cpu()  # Ensure last_value is on CPU for consistency

        last_gae_lam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                # Use last_value_cpu for the value after the last step
                next_non_terminal = 1.0 - self.dones[t].float()  # self.dones is on CPU
                next_values = last_value_cpu
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = cpu_values[t + 1]  # Value of the actual next step

            # Use cpu_values tensor for V(s_t)
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - cpu_values[t]
            last_gae_lam = delta + gamma * lambda_ * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam  # Assign to the 1D tensor
            returns[t] = advantages[t] + cpu_values[t]  # Return = Advantage + Value

        # Store computed tensors in the buffer object
        self.returns = returns
        self.advantages = advantages

        return returns, advantages  # Optional: return them as well

    def get_batch(self, batch_size):
        """ Returns shuffled minibatches from the stored data. """
        n_samples = len(self.states) # Use length of states/actions list
        if n_samples == 0:
             # Handle case where buffer might be empty after clear or before first fill
             print("Warning: get_batch called on empty buffer.")
             return # Yield nothing

        indices = np.random.permutation(n_samples)

        # --- Corrected Stacking/Usage ---
        # Stack only the lists of tensors (these should be on CPU)
        all_states = torch.stack(self.states)
        all_actions = torch.stack(self.actions)
        all_log_probs = torch.stack(self.log_probs)

        # Use the pre-computed tensors directly (they should be on CPU)
        # Add a check to ensure compute_returns_and_advantages was called
        if not hasattr(self, 'returns') or not hasattr(self, 'advantages'):
             raise RuntimeError("compute_returns_and_advantages must be called before get_batch")
        # These are already tensors, no need to stack
        all_returns = self.returns
        all_advantages = self.advantages
        # --------------------------

        # Basic shape consistency check
        if not (all_states.shape[0] == n_samples and \
                all_actions.shape[0] == n_samples and \
                all_log_probs.shape[0] == n_samples and \
                all_returns.shape[0] == n_samples and \
                all_advantages.shape[0] == n_samples):
            raise RuntimeError(f"Shape mismatch in buffer data! Expected {n_samples} samples.")


        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield (
                all_states[batch_indices],    # Shape: (batch_size, latent_dim)
                all_actions[batch_indices],   # Shape: (batch_size, action_dim)
                all_log_probs[batch_indices], # Shape: (batch_size,)
                all_returns[batch_indices],   # Shape: (batch_size,)
                all_advantages[batch_indices],# Shape: (batch_size,)
            )

    def clear(self):
        self.__init__() # Reset all lists

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
        actor.train() # Set to train mode for updates
        critic.train()
        approx_kl_divs = []

        for epoch in range(EPOCHS_PER_UPDATE):
            for states, actions, old_log_probs, returns, advantages in buffer.get_batch(MINIBATCH_SIZE):
                states, actions, old_log_probs, returns, advantages = \
                    states.to(DEVICE), actions.to(DEVICE), old_log_probs.to(DEVICE), \
                    returns.to(DEVICE), advantages.to(DEVICE)

                # Detach tensors coming from the buffer that shouldn't propagate gradients from the rollout phase
                # Actions are needed for log_prob calculation but should be treated as fixed inputs here
                # Old log probs are fixed targets for the ratio calculation
                # Returns and Advantages are targets/weights for the losses
                actions = actions.detach()
                old_log_probs = old_log_probs.detach()
                returns = returns.detach()
                advantages = advantages.detach()  # Already normalized, treat as constant weights

                # --- Calculate Actor Loss ---
                new_dist = actor(states)
                # Calculate log_prob using the detached actions from the buffer
                new_log_probs = new_dist.log_prob(actions).sum(1)  # Sum across action dims
                entropy = new_dist.entropy().sum(1).mean()  # Mean entropy

                # Calculate ratio r_t(theta) = exp(log pi_new - log pi_old)
                # old_log_probs is now detached
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped Surrogate Objective (advantages is detached)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()  # Negative because we minimize

                # --- Calculate Critic Loss ---
                new_values = critic(states).squeeze(1)  # Shape: (batch_size,)
                # Simple value loss: MSE against calculated returns (returns is detached)
                critic_loss = F.mse_loss(new_values, returns)

                # --- Calculate Total Loss ---
                loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy

                # --- Compute Gradients (Once using combined loss) ---
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()  # Should now work correctly

                # --- Actor Update ---
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
                actor_optimizer.step()

                # --- Critic Update ---
                nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
                critic_optimizer.step()

                # --- KL Divergence Tracking (Optional) ---
                with torch.no_grad():
                    approx_kl = (ratio - 1) - log_ratio # http://joschu.net/blog/kl-approx.html
                    approx_kl = approx_kl.mean().item()
                    approx_kl_divs.append(approx_kl)

            # Optional: Early stopping based on KL divergence
            if TARGET_KL is not None and np.mean(approx_kl_divs[-len(buffer.states)//MINIBATCH_SIZE:]) > TARGET_KL:
                print(f"  Epoch {epoch+1}: Early stopping at KL divergence {np.mean(approx_kl_divs):.4f} > {TARGET_KL}")
                break


        print(f"Update {update}, Optimizing for {epoch+1} epochs. Mean KL: {np.mean(approx_kl_divs):.4f}")
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