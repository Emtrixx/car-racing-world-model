import torch
import torch.nn as nn  # For type hinting
import torch.optim as optim  # For type hinting
import torch.nn.functional as F
import numpy as np


# from utils import RolloutBuffer, PPOHyperparameters # If defined in utils.py

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