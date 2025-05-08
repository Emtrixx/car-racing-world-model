from dataclasses import dataclass

import torch
import torch.nn as nn  # For type hinting
import torch.optim as optim  # For type hinting
import torch.nn.functional as F
import numpy as np

from projects.gym_stuff.car_racing.utils import ENV_NAME, LATENT_DIM


@dataclass
class PPOHyperparameters:
    gamma: float = 0.99
    lambda_gae: float = 0.95 # Renamed from LAMBDA to avoid keyword clash
    epsilon_clip: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs_per_update: int = 10
    minibatch_size: int = 64
    # steps_per_batch: int # This is context-dependent (real vs dream)
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip_norm: float = 0.5
    target_kl: float = 0.015


PPO_DREAM_ACTOR_SAVE_FILENAME = f"{ENV_NAME}_ppo_dream_actor_ld{LATENT_DIM}.pth"
PPO_DREAM_CRITIC_SAVE_FILENAME = f"{ENV_NAME}_ppo_dream_critic_ld{LATENT_DIM}.pth"
PPO_ACTOR_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_actor_ld{LATENT_DIM}.pth"
PPO_CRITIC_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_critic_ld{LATENT_DIM}.pth"


def perform_ppo_update(
        buffer,  # Should be an instance of RolloutBuffer
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        hyperparams,  # Should be an instance of PPOHyperparameters or similar dict/object
        device: torch.device,
        last_value_for_gae: torch.Tensor  # V(s_T) for the batch, already on correct device
):
    """
    Performs PPO optimization epochs on the data in the buffer.

    Args:
        buffer (RolloutBuffer): Buffer containing collected trajectory data.
        actor (nn.Module): The actor network.
        critic (nn.Module): The critic network.
        actor_optimizer (optim.Optimizer): Optimizer for the actor.
        critic_optimizer (optim.Optimizer): Optimizer for the critic.
        hyperparams (PPOHyperparameters): Dataclass or object holding PPO hyperparameters.
        device (torch.device): Device to perform computations on.
        last_value_for_gae (torch.Tensor): Value of the state after the last step in the buffer,
                                           used for GAE calculation. Should be on `device`.

    Returns:
        float: Mean approximate KL divergence over the update epochs.
    """
    actor.train()
    critic.train()
    approx_kl_divs_all_epochs = []

    # Compute returns and advantages using GAE
    # last_value_for_gae is now passed as an argument
    buffer.compute_returns_and_advantages(last_value_for_gae, hyperparams.gamma, hyperparams.lambda_gae)

    # Normalize advantages (important for stability)
    buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    for epoch in range(hyperparams.epochs_per_update):
        approx_kl_divs_this_epoch = []
        for states, actions, old_log_probs, returns, advantages in buffer.get_batch(hyperparams.minibatch_size):
            states, actions, old_log_probs, returns, advantages = \
                states.to(device), actions.to(device), old_log_probs.to(device), \
                    returns.to(device), advantages.to(device)

            actions, old_log_probs, returns, advantages = \
                actions.detach(), old_log_probs.detach(), returns.detach(), advantages.detach()

            # --- Calculate Actor Loss ---
            new_dist = actor(states)
            new_log_probs = new_dist.log_prob(actions).sum(1)
            entropy = new_dist.entropy().sum(1).mean()
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - hyperparams.epsilon_clip, 1.0 + hyperparams.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Calculate Critic Loss ---
            new_values = critic(states).squeeze(1)
            critic_loss = F.mse_loss(new_values, returns)

            # --- Calculate Total Loss ---
            loss = actor_loss + hyperparams.vf_coef * critic_loss - hyperparams.entropy_coef * entropy

            # --- Compute Gradients ---
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()

            # --- Actor Update ---
            if hyperparams.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), hyperparams.grad_clip_norm)
            actor_optimizer.step()

            # --- Critic Update ---
            if hyperparams.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(critic.parameters(), hyperparams.grad_clip_norm)
            critic_optimizer.step()

            with torch.no_grad():
                approx_kl = (ratio - 1) - log_ratio  # http://joschu.net/blog/kl-approx.html
                approx_kl = approx_kl.mean().item()
                approx_kl_divs_this_epoch.append(approx_kl)

        epoch_mean_kl = np.mean(approx_kl_divs_this_epoch)
        approx_kl_divs_all_epochs.append(epoch_mean_kl)
        if hyperparams.target_kl is not None and epoch_mean_kl > hyperparams.target_kl:
            # print(f"  PPO Epoch {epoch+1}: Early stopping at KL divergence {epoch_mean_kl:.4f} > {hyperparams.target_kl}")
            break

    return np.mean(approx_kl_divs_all_epochs)


# --- Simple Random Policy ---
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state): # state can be observation or latent state, ignored here
        return self.action_space.sample()


class PPOPolicyWrapper:
    def __init__(self, actor_model, device, deterministic=False, action_space_low=None, action_space_high=None):
        self.actor_model = actor_model
        self.device = device
        self.deterministic = deterministic # False for exploration, True for exploitation/eval

        if action_space_low is None:
            # Defaults for CarRacing-v3
            action_space_low = [-1.0, 0.0, 0.0]
        if action_space_high is None:
            action_space_high = [1.0, 1.0, 1.0]

        self.action_space_low_tensor = torch.tensor(action_space_low, device=self.device, dtype=torch.float32)
        self.action_space_high_tensor = torch.tensor(action_space_high, device=self.device, dtype=torch.float32)


    def get_action(self, z_t_numpy): # Expects a numpy array for z_t
        self.actor_model.eval() # Ensure actor is in eval mode
        z_t = torch.tensor(z_t_numpy, dtype=torch.float32).to(self.device).unsqueeze(0) # Add batch dim

        with torch.no_grad():
            dist = self.actor_model(z_t) # actor_model should be the loaded Actor network
            if self.deterministic:
                action_raw = dist.mean
            else:
                action_raw = dist.sample() # Sample for exploration

            # Process action (same logic as in train_ppo.py and play_game.py)
            # Steering is output by actor's fc_mean in tanh range already for its first component
            # Gas/Brake means are unbounded from fc_mean, then dist samples.
            # We apply tanh to the sample, then scale.

            action_processed = torch.tanh(action_raw) # Squash sample to [-1, 1]

            action_scaled = torch.zeros_like(action_processed)
            action_scaled[:, 0] = action_processed[:, 0] # Steering: directly use tanh output
            action_scaled[:, 1:] = (action_processed[:, 1:] + 1.0) / 2.0 # Gas, Brake: scale from [-1,1] to [0,1]

            action_clipped = torch.clamp(action_scaled,
                                         self.action_space_low_tensor,
                                         self.action_space_high_tensor)

        return action_clipped.squeeze(0).cpu().numpy()

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
        # These will be populated by compute_returns_and_advantages
        self.returns = None     # torch.Tensor
        self.advantages = None  # torch.Tensor

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value_tensor, gamma, lambda_gae):
        n_steps = len(self.rewards)
        if n_steps == 0:
            self.returns = torch.tensor([])
            self.advantages = torch.tensor([])
            return

        # Ensure last_value_tensor is on CPU if other tensors are
        last_value_tensor = last_value_tensor.cpu()

        advantages_list = [torch.zeros_like(self.rewards[0])] * n_steps  # Temp list
        returns_list = [torch.zeros_like(self.rewards[0])] * n_steps  # Temp list

        # Stack values for easier access and ensure it's on CPU
        cpu_values = torch.stack(self.values)  # self.values are stored on CPU

        last_gae_lam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t].float()  # self.dones is on CPU
                next_values = last_value_tensor  # Value after the last collected step
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = cpu_values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - cpu_values[t]
            last_gae_lam = delta + gamma * lambda_gae * next_non_terminal * last_gae_lam
            advantages_list[t] = last_gae_lam
            returns_list[t] = advantages_list[t] + cpu_values[t]

        self.returns = torch.stack(returns_list)
        self.advantages = torch.stack(advantages_list)

        # Normalize advantages (important for stability)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

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
